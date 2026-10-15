# An OpenAI-compatible server for Qwen3.5 / 3.6 / 3.8 GGUFs (streaming chat completions with tool calls), for harnesses like opencode or pi.
#
# Usage:
#   LUMINAL_GGUF_NATIVE=1 julia -t auto,1 --project=. examples/qwen35_server.jl <model.gguf> [--mtp=<gguf with the MTP head>] [--port=8080]
#       [--host=127.0.0.1] [--max-seq=16384] [--int8] [--no-spec] [--name=<model id>] [--min-avail=20]
# Then point the harness at http://localhost:8080/v1 (model id: --name, default the file's name).
#
# Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions ("stream": true gives server-sent events). Request fields used: messages
# (system / user / assistant with tool_calls / tool), tools, stream, max_tokens (max_completion_tokens), temperature, top_p, top_k,
# reasoning_effort ("medium", "low", "xhigh"), chat_template_kwargs.enable_thinking, stop (strings), seed. Defaults are Qwen's recommended
# coding settings: temperature 1.0, top_p 0.95, top_k 20, medium reasoning. The reply's thinking is `reasoning_content`, its text `content`,
# and <tool_call> blocks become OpenAI `tool_calls`.
#
# One GPU request at a time: all GPU work runs on the main task; HTTP handlers queue jobs for it and stream its events back.
using Luminal, Luminal.NN
using HTTP, JSON3, Random, Printf, Dates

const LT = Luminal.LlamaTokenization

# ---------------------------------------------------------------------------------------------------------------------------------
# JSON the way Python's json.dumps writes it (the chat template's `tojson` does: ", " and ": " separators, no ASCII escaping)

pyjson(x::Nothing) = "null"
pyjson(x::Bool) = x ? "true" : "false"
pyjson(x::Number) = JSON3.write(x)
pyjson(x::AbstractString) = JSON3.write(String(x))
pyjson(x::AbstractVector) = "[" * join((pyjson(v) for v in x), ", ") * "]"
pyjson(x::AbstractDict) = "{" * join((JSON3.write(String(k)) * ": " * pyjson(v) for (k, v) in x), ", ") * "}"

# ---------------------------------------------------------------------------------------------------------------------------------
# The chat template (Qwen3.8's, as stored in the GGUF) for OpenAI-style messages

const XHIGH_TEXT = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
const LOW_TEXT = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
const TOOLS_TAIL = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

field(m, k::Symbol, default=nothing) = (v = get(m, k, default); v === nothing ? default : v)

function content_text(c)
    c === nothing && return ""
    c isa AbstractString && return String(c)
    c isa AbstractVector && return join((String(field(it, :text, "")) for it in c if field(it, :text) !== nothing), "")
    error("Unexpected content type.")
end

function render_chat(messages, tools; thinking::Bool=true, effort::String="medium")
    isempty(messages) && error("No messages provided.")
    io = IOBuffer()
    nsys = 0; sys = String[]
    for m in messages
        String(m.role) in ("system", "developer") || break
        c = strip(content_text(field(m, :content)))
        isempty(c) || push!(sys, c)
        nsys += 1
    end
    merged = join(sys, "\n")
    effort = effort == "high" ? "xhigh" : effort
    effort in ("xhigh", "medium", "low") || error("Unexpected reasoning effort $effort. Supported types are xhigh (default), medium, and low.")
    ri = !thinking ? "" : effort == "xhigh" ? XHIGH_TEXT : effort == "low" ? LOW_TEXT : ""
    if tools !== nothing && !isempty(tools)
        print(io, "<|im_start|>system\n")
        isempty(ri) || print(io, ri, "\n\n")
        print(io, "# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for t in tools
            print(io, "\n", pyjson(t))
        end
        print(io, "\n</tools>", TOOLS_TAIL)
        isempty(merged) || print(io, "\n\n", merged)
        print(io, "<|im_end|>\n")
    elseif !isempty(merged)
        print(io, "<|im_start|>system\n", isempty(ri) ? "" : ri * "\n\n", merged, "<|im_end|>\n")
    elseif !isempty(ri)
        print(io, "<|im_start|>system\n", ri, "<|im_end|>\n")
    end
    n = length(messages)
    for i in nsys+1:n
        m = messages[i]; role = String(m.role)
        content = strip(content_text(field(m, :content)))
        if role in ("system", "developer")
            error("System message must be at the beginning.")
        elseif role == "user"
            print(io, "<|im_start|>user\n", content, "<|im_end|>\n")
        elseif role == "assistant"
            rc = field(m, :reasoning_content)
            rc = rc isa AbstractString ? strip(rc) : ""
            print(io, "<|im_start|>assistant\n<think>\n", rc, "\n</think>\n\n", content)
            tcs = field(m, :tool_calls)
            if tcs !== nothing
                for (j, tc) in enumerate(tcs)
                    fn = field(tc, Symbol("function"), tc)
                    name = field(fn, :name)
                    name === nothing && error("Tool call is missing a function name.")
                    print(io, j == 1 ? (isempty(content) ? "" : "\n\n") : "\n", "<tool_call>\n<function=", name, ">\n")
                    args = field(fn, :arguments)
                    args isa AbstractString && (args = isempty(strip(args)) ? nothing : JSON3.read(args))
                    if args !== nothing
                        for (k, v) in pairs(args)
                            print(io, "<parameter=", k, ">\n", v isa AbstractString ? v : pyjson(v), "\n</parameter>\n")
                        end
                    end
                    print(io, "</function>\n</tool_call>")
                end
            end
            print(io, "<|im_end|>\n")
        elseif role == "tool"
            (i == nsys + 1 || String(messages[i-1].role) != "tool") && print(io, "<|im_start|>user")
            print(io, "\n<tool_response>\n", content, "\n</tool_response>")
            (i == n || String(messages[i+1].role) != "tool") && print(io, "<|im_end|>\n")
        else
            error("Unexpected message role.")
        end
    end
    print(io, "<|im_start|>assistant\n", thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
    return String(take!(io))
end

# ---------------------------------------------------------------------------------------------------------------------------------
# The model's reply -> reasoning / content / tool calls, incrementally. Text that might be the start of a tag (or trailing white space before
# one) is held back until it is decided.

mutable struct OutParser
    mode::Symbol                 # :reasoning, :content, :toolcall
    buf::String
    lead::Bool                   # drop white space at the start of the content (the template puts "\n\n" after </think>)
    schema::Dict{String,Dict{String,String}}    # tool name -> parameter name -> JSON type
    ncalls::Int
end
function OutParser(thinking::Bool, tools)
    schema = Dict{String,Dict{String,String}}()
    for t in something(tools, [])
        fn = field(t, Symbol("function"), t); name = field(fn, :name); name === nothing && continue
        props = field(field(fn, :parameters, Dict()), :properties, Dict())
        schema[String(name)] = Dict(String(k) => String(field(v, :type, "")) for (k, v) in pairs(props))
    end
    OutParser(thinking ? :reasoning : :content, "", true, schema, 0)
end

partial_suffix(s::AbstractString, tag::AbstractString) = (for j in min(length(tag) - 1, length(s)):-1:1; endswith(s, tag[1:j]) && return j; end; 0)

function tool_arguments(p::OutParser, name::String, body::String)
    types = get(p.schema, name, Dict{String,String}())
    args = Pair{String,Any}[]
    for m in eachmatch(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>"s, body)
        k, v = String(m[1]), String(m[2])
        val = v
        if get(types, k, "") != "string"
            try; val = JSON3.read(v); catch; end
        end
        push!(args, k => val)
    end
    return "{" * join((JSON3.write(k) * ":" * (v isa AbstractString ? JSON3.write(String(v)) : JSON3.write(v)) for (k, v) in args), ",") * "}"
end

# -> a vector of (:reasoning | :content, text) and (:tool_call, (name, arguments_json))
function feed!(p::OutParser, text::AbstractString; final::Bool=false)
    p.buf *= text
    ev = Tuple{Symbol,Any}[]
    while true
        if p.mode == :reasoning
            i = findfirst("</think>", p.buf)
            if i !== nothing
                r = p.buf[1:first(i)-1]; isempty(r) || push!(ev, (:reasoning, r))
                p.buf = p.buf[last(i)+1:end]; p.mode = :content; p.lead = true
                continue
            end
            k = final ? 0 : partial_suffix(p.buf, "</think>")
            cut = lastindex(p.buf) - k
            r = p.buf[1:cut]; isempty(r) || push!(ev, (:reasoning, r))
            p.buf = p.buf[cut+1:end]
            break
        elseif p.mode == :content
            p.lead && (p.buf = lstrip(p.buf); isempty(p.buf) || (p.lead = false))
            i = findfirst("<tool_call>", p.buf)
            if i !== nothing
                c = rstrip(p.buf[1:first(i)-1]); isempty(c) || push!(ev, (:content, String(c)))
                p.buf = p.buf[last(i)+1:end]; p.mode = :toolcall
                continue
            end
            k = final ? 0 : partial_suffix(p.buf, "<tool_call>")
            head = p.buf[1:lastindex(p.buf)-k]
            c = final ? head : rstrip(head)                       # white space before a (possible) tag stays held back
            isempty(c) || push!(ev, (:content, String(c)))
            p.buf = p.buf[ncodeunits(c)+1:end]
            break
        else
            i = findfirst("</tool_call>", p.buf)
            if i === nothing
                if final && !isempty(strip(p.buf))                      # unfinished: hand the text back as content
                    push!(ev, (:content, "<tool_call>" * p.buf)); p.buf = ""
                end
                break
            end
            body = p.buf[1:first(i)-1]
            p.buf = p.buf[last(i)+1:end]; p.mode = :content; p.lead = true
            m = match(r"<function=([^>\n]+)>"s, body)
            if m === nothing
                push!(ev, (:content, "<tool_call>" * body * "</tool_call>"))
            else
                name = String(m[1]); p.ncalls += 1
                push!(ev, (:tool_call, (name, tool_arguments(p, name, body))))
            end
            continue
        end
    end
    return ev
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Jobs: handlers -> the GPU worker (main task) -> events back

struct Job
    prompt::String
    max_tokens::Int
    sampler::Any
    tools::Any
    thinking::Bool
    stop::Vector{String}
    events::Channel{Any}
    cancel::Base.RefValue{Bool}
end
const JOBS = Channel{Job}(64)

function run_job!(s, d, tok, job::Job, use_spec::Bool)
    ids = Int.(LT.encode(tok, job.prompt; bos = false))
    t0 = time()
    if length(ids) + job.max_tokens + 8 > s.max_seq
        put!(job.events, (:error, "the prompt has $(length(ids)) tokens and max_tokens is $(job.max_tokens); the server's context is $(s.max_seq)"))
        return
    end
    parser = OutParser(job.thinking, job.tools)
    gen = Int[]; sent = 0                        # bytes of the decoded text already fed to the parser
    reason = :stop; visible = ""
    eos = Set(tok.eos_ids)
    t_first = Ref(0.0)
    function emit(text; final=false)
        for e in feed!(parser, text; final)
            put!(job.events, e)
            e[1] == :content && (visible *= e[2])
        end
    end
    function on_token(new)
        for t in new
            if t in eos; reason = :stop; return true; end
            push!(gen, t)
            t_first[] == 0.0 && (t_first[] = time())
            if length(gen) >= job.max_tokens; reason = :length; break; end
        end
        job.cancel[] && (reason = :cancel; return true)
        text = LT.decode(tok, gen)
        endswith(text, '�') && reason != :length && return false           # an incomplete UTF-8 sequence: wait for its rest
        emit(text[sent+1:end]); sent = ncodeunits(text)
        if !isempty(job.stop) && any(st -> occursin(st, visible), job.stop)
            reason = :stop; return true
        end
        return reason == :length
    end
    try
        if use_spec
            generate_ids_speculative(s, d, ids; max_new_tokens = job.max_tokens + 8, ndraft = 0, sampler = job.sampler, on_token)
        else
            generate_ids(s, [ids]; max_new_tokens = job.max_tokens + 8, sampler = job.sampler, on_token)
        end
        text = LT.decode(tok, gen)
        emit(text[sent+1:end]; final = true)
        length(gen) >= job.max_tokens && reason == :stop && !isempty(gen) && !(gen[end] in eos) && (reason = :length)
        t1 = time()
        put!(job.events, (:done, (finish = parser.ncalls > 0 && reason == :stop ? "tool_calls" : reason == :length ? "length" : "stop",
                                  prompt_tokens = length(ids), completion_tokens = length(gen),
                                  prefill_s = (t_first[] == 0.0 ? t1 : t_first[]) - t0, total_s = t1 - t0)))
    catch err
        @error "generation failed" exception = (err, catch_backtrace())
        put!(job.events, (:error, sprint(showerror, err)))
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# HTTP

json_response(http, status, obj) = begin
    body = JSON3.write(obj)
    HTTP.setstatus(http, status)
    HTTP.setheader(http, "Content-Type" => "application/json"); HTTP.setheader(http, "Content-Length" => string(sizeof(body)))
    HTTP.startwrite(http); write(http, body)
end
err_response(http, status, msg) = json_response(http, status, (error = (message = msg, type = "invalid_request_error", code = status),))

function chat_handler(http::HTTP.Stream, name::String)
    req = JSON3.read(String(read(http)))
    messages = field(req, :messages)
    messages === nothing && return err_response(http, 400, "messages is required")
    tools = field(req, :tools)
    kwargs = field(req, :chat_template_kwargs, Dict())
    thinking = field(kwargs, :enable_thinking, true) !== false
    effort = String(something(field(req, :reasoning_effort), field(kwargs, :reasoning_effort), "medium"))
    prompt = try
        render_chat(messages, tools; thinking, effort)
    catch e
        return err_response(http, 400, sprint(showerror, e))
    end
    temperature = Float64(field(req, :temperature, 1.0))
    seed = field(req, :seed)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(Int(seed))
    sampler = temperature <= 0 ? nothing :
        Sampler(temperature = temperature, top_p = Float64(field(req, :top_p, 0.95)), top_k = Int(field(req, :top_k, 20)), rng = rng)
    stop = field(req, :stop, String[]); stop = stop isa AbstractString ? [String(stop)] : String.(stop)
    max_tokens = Int(something(field(req, :max_completion_tokens), field(req, :max_tokens), 4096))
    job = Job(prompt, max_tokens, sampler, tools, thinking, stop, Channel{Any}(Inf), Ref(false))
    put!(JOBS, job)
    id = "chatcmpl-" * randstring(16); created = Int(floor(time()))
    stream = field(req, :stream, false) === true
    chunk(delta; finish = nothing, usage = nothing) = (c = (id = id, object = "chat.completion.chunk", created = created, model = name,
                                                            choices = [(index = 0, delta = delta, finish_reason = finish)]);
                                                       usage === nothing ? c : merge(c, (usage = usage,)))
    if stream
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream"); HTTP.setheader(http, "Cache-Control" => "no-cache")
        HTTP.startwrite(http)
        sse(obj) = write(http, "data: ", obj isa AbstractString ? obj : JSON3.write(obj), "\n\n")
        try
            sse(chunk((role = "assistant", content = "")))
            ncall = 0
            while true
                kind, val = take!(job.events)
                if kind == :reasoning
                    sse(chunk((reasoning_content = val,)))
                elseif kind == :content
                    sse(chunk((content = val,)))
                elseif kind == :tool_call
                    sse(chunk((tool_calls = [(index = ncall, id = "call_" * randstring(12), type = "function",
                                              var"function" = (name = val[1], arguments = val[2]))],)))
                    ncall += 1
                elseif kind == :done
                    u = (prompt_tokens = val.prompt_tokens, completion_tokens = val.completion_tokens, total_tokens = val.prompt_tokens + val.completion_tokens)
                    sse(chunk((;); finish = val.finish, usage = u))
                    sse("[DONE]")
                    @info @sprintf("%d prompt tokens (first token after %.2f s), %d generated at %.1f tok/s", val.prompt_tokens, val.prefill_s,
                                   val.completion_tokens, val.completion_tokens / max(val.total_s - val.prefill_s, 1e-6))
                    break
                else
                    sse((error = (message = val, type = "server_error"),)); sse("[DONE]"); break
                end
            end
        catch e
            job.cancel[] = true                                              # the client went away: stop generating
        end
    else
        reasoning = IOBuffer(); content = IOBuffer(); calls = Any[]; fin = nothing; usage = nothing
        while true
            kind, val = take!(job.events)
            kind == :reasoning && print(reasoning, val)
            kind == :content && print(content, val)
            kind == :tool_call && push!(calls, (id = "call_" * randstring(12), type = "function", var"function" = (name = val[1], arguments = val[2])))
            if kind == :done
                fin = val.finish; usage = (prompt_tokens = val.prompt_tokens, completion_tokens = val.completion_tokens, total_tokens = val.prompt_tokens + val.completion_tokens); break
            elseif kind == :error
                return err_response(http, 500, val)
            end
        end
        msg = (role = "assistant", content = String(take!(content)), reasoning_content = String(take!(reasoning)))
        isempty(calls) || (msg = merge(msg, (tool_calls = calls,)))
        json_response(http, 200, (id = id, object = "chat.completion", created = created, model = name,
                                  choices = [(index = 0, message = msg, finish_reason = fin)], usage = usage))
    end
end

function route(http::HTTP.Stream, name::String)
    req = http.message
    path = HTTP.URIs.URI(req.target).path
    try
        if req.method == "GET" && path in ("/health", "/v1/health")
            json_response(http, 200, (status = "ok",))
        elseif req.method == "GET" && path in ("/v1/models", "/models")
            json_response(http, 200, (object = "list", data = [(id = name, object = "model", created = 0, owned_by = "luminal")]))
        elseif req.method == "POST" && path in ("/v1/chat/completions", "/chat/completions")
            chat_handler(http, name)
        else
            err_response(http, 404, "not found: $(req.method) $path")
        end
    catch e
        e isa Base.IOError || e isa HTTP.Exceptions.HTTPError || @error "request failed" exception = (e, catch_backtrace())
        try; err_response(http, 500, sprint(showerror, e)); catch; end
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(n, d) = (i = findfirst(a -> startswith(a, "--$n="), ARGS); i === nothing ? d : ARGS[i][length(n)+4:end])
    isempty(args) && error("usage: qwen35_server.jl <model.gguf> [--mtp=<gguf>] [--port=8080] [--host=127.0.0.1] [--max-seq=16384] [--int8] [--no-spec] [--name=id]")
    base = args[1]; mtp = flag("mtp", base)
    port = parse(Int, flag("port", "8080")); host = flag("host", "127.0.0.1")
    use_spec = !("--no-spec" in ARGS)
    name = flag("name", splitext(basename(base))[1])
    T = "--int8" in ARGS ? Int8 : Luminal.Int4
    f = read_gguf(base); cfg = qwen35_config(f); tok = LlamaTokenizer(f)
    @info "loading $base"
    W = load_qwen35_gguf(base, cfg; quantize = _ -> T)
    use_spec && merge!(W, load_qwen35_mtp(mtp, cfg; quantize = _ -> T, block = cfg.n_layers))
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = parse(Int, flag("max-seq", "16384")), weights = W, decode_weights = T,
                     kv_dtype = Float16)
    d = use_spec ? MTPDrafter(s) : nothing
    @info "warming up (compiles the graphs)"
    wid = Int.(LT.encode(tok, render_chat([(role = "user", content = "Say hello.")], nothing); bos = false))
    for sm in (nothing, Sampler(temperature = 1.0, top_p = 0.95, top_k = 20))
        use_spec ? (generate_ids_speculative(s, d, wid; max_new_tokens = 24, ndraft = 0, sampler = sm); generate_ids_speculative(s, d, wid; max_new_tokens = 24, ndraft = 0, sampler = sm)) :
                   generate_ids(s, [wid]; max_new_tokens = 8, sampler = sm)
    end
    server = HTTP.serve!(http -> route(http, name), host, port; stream = true)
    @info "serving $name on http://$host:$port/v1 (speculative decoding: $use_spec, context $(s.max_seq))"
    while true
        job = take!(JOBS)
        job.cancel[] && continue
        run_job!(s, d, tok, job, use_spec)
        Luminal.reclaim!(s.device)
    end
end

(abspath(PROGRAM_FILE) == (@__FILE__)) && main()
