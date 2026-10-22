# An OpenAI-compatible server for Qwen3.5 / 3.6 / 3.8 GGUFs (streaming chat completions with tool calls), for harnesses like opencode or pi.
#
# Usage:
#   LUMINAL_GGUF_NATIVE=1 julia -t auto,1 --project=. examples/qwen35_server.jl <model.gguf> [--mtp=<gguf with the MTP head>] [--port=8080]
#       [--host=127.0.0.1] [--max-seq=16384] [--int8] [--kv-int8] [--no-spec] [--dflash=<DFlash2 draft gguf>] [--name=<model id>] [--min-avail=20] [--reasoning-budget=N]
# Then point the harness at http://localhost:8080/v1 (model id: --name, default the file's name).
#
# Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions ("stream": true gives server-sent events). Request fields used: messages
# (system / user / assistant with tool_calls / tool), tools, stream, max_tokens (max_completion_tokens), temperature, top_p, top_k,
# reasoning_effort ("medium", "low", "xhigh"), thinking_budget (or thinking.budget_tokens, reasoning_budget: reasoning tokens before the thinking is cut off), chat_template_kwargs.enable_thinking, stop (strings), seed. Defaults are Qwen's recommended
# coding settings: temperature 1.0, top_p 0.95, top_k 20, medium reasoning. The reply's thinking is `reasoning_content`, its text `content`,
# and <tool_call> blocks become OpenAI `tool_calls`.
#
# One GPU request at a time: all GPU work runs in one task pinned off the main thread (hence -t auto,1); HTTP handlers queue jobs for it and stream its events back.
using Luminal, Luminal.NN
using HTTP, JSON3, Random, Printf, Dates, Logging

const LT = Luminal.LlamaTokenization
# HTTP.jl logs "handle_connection handler error ... EOFError" (a full stack trace) whenever a client closes a connection mid-request, which
# opencode does on every abort or timeout; that is not an error of the server, so those records are dropped.
struct QuietHTTP <: Logging.AbstractLogger
    inner::Logging.AbstractLogger
end
Logging.min_enabled_level(l::QuietHTTP) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::QuietHTTP, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::QuietHTTP) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::QuietHTTP, level, message, _module, group, id, file, line; kwargs...)
    msg = string(message)
    occursin("handle_connection handler error", msg) && occursin("EOFError", msg) && return nothing
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end

logline(msg) = (println(stderr, Dates.format(now(), "HH:MM:SS "), msg); flush(stderr))

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

function render_chat(messages, tools; thinking::Bool=true, effort::String="medium", parts=nothing)
    isempty(messages) && error("No messages provided.")
    io = IOBuffer()
    # `parts`, a Dict{Symbol,IOBuffer}, receives the same text by kind (for the per-request token breakdown)
    pr(kind::Symbol, xs...) = (print(io, xs...); parts === nothing || print(get!(parts, kind, IOBuffer()), xs...))
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
        pr(:markup, "<|im_start|>system\n")
        isempty(ri) || pr(:system, ri, "\n\n")
        pr(:tools, "# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for t in tools
            pr(:tools, "\n", pyjson(t))
        end
        pr(:tools, "\n</tools>", TOOLS_TAIL)
        isempty(merged) || pr(:system, "\n\n", merged)
        pr(:markup, "<|im_end|>\n")
    elseif !isempty(merged)
        pr(:markup, "<|im_start|>system\n"); pr(:system, isempty(ri) ? "" : ri * "\n\n", merged); pr(:markup, "<|im_end|>\n")
    elseif !isempty(ri)
        pr(:markup, "<|im_start|>system\n"); pr(:system, ri); pr(:markup, "<|im_end|>\n")
    end
    n = length(messages)
    for i in nsys+1:n
        m = messages[i]; role = String(m.role)
        content = strip(content_text(field(m, :content)))
        if role in ("system", "developer")
            error("System message must be at the beginning.")
        elseif role == "user"
            pr(:markup, "<|im_start|>user\n"); pr(:user, content); pr(:markup, "<|im_end|>\n")
        elseif role == "assistant"
            rc = field(m, :reasoning_content)
            rc = rc isa AbstractString ? strip(rc) : ""
            pr(:markup, "<|im_start|>assistant\n<think>\n"); pr(:reasoning, rc); pr(:markup, "\n</think>\n\n"); pr(:assistant, content)
            tcs = field(m, :tool_calls)
            if tcs !== nothing
                for (j, tc) in enumerate(tcs)
                    fn = field(tc, Symbol("function"), tc)
                    name = field(fn, :name)
                    name === nothing && error("Tool call is missing a function name.")
                    pr(:toolcalls, j == 1 ? (isempty(content) ? "" : "\n\n") : "\n", "<tool_call>\n<function=", name, ">\n")
                    args = field(fn, :arguments)
                    args isa AbstractString && (args = isempty(strip(args)) ? nothing : JSON3.read(args))
                    if args !== nothing
                        for (k, v) in pairs(args)
                            pr(:toolcalls, "<parameter=", k, ">\n", v isa AbstractString ? v : pyjson(v), "\n</parameter>\n")
                        end
                    end
                    pr(:toolcalls, "</function>\n</tool_call>")
                end
            end
            pr(:markup, "<|im_end|>\n")
        elseif role == "tool"
            (i == nsys + 1 || String(messages[i-1].role) != "tool") && pr(:markup, "<|im_start|>user")
            pr(:markup, "\n<tool_response>\n"); pr(:toolresults, content); pr(:markup, "\n</tool_response>")
            (i == n || String(messages[i+1].role) != "tool") && pr(:markup, "<|im_end|>\n")
        else
            error("Unexpected message role.")
        end
    end
    pr(:markup, "<|im_start|>assistant\n", thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
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
    tstate::Symbol               # inside a tool call: :head (before <function=>), :between (parameters), :value_str (streaming a string value), :value_hold (a typed value, emitted whole)
    tname::String                # the tool being called
    tparam::String               # the parameter being read
    tfirst::Bool                 # no parameter written yet (no comma before the next one)
    topen::Bool                  # the call's name has been sent and its arguments are not closed yet
    nargs::Int                   # characters of arguments sent for the current call
end
function OutParser(thinking::Bool, tools)
    schema = Dict{String,Dict{String,String}}()
    for t in something(tools, [])
        fn = field(t, Symbol("function"), t); name = field(fn, :name); name === nothing && continue
        props = field(field(fn, :parameters, Dict()), :properties, Dict())
        schema[String(name)] = Dict(String(k) => String(field(v, :type, "")) for (k, v) in pairs(props))
    end
    OutParser(thinking ? :reasoning : :content, "", true, schema, 0, :head, "", "", true, false, 0)
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

# s cut after its first n bytes (n is the end of a character: the held-back suffix is ASCII)
split_bytes(s::String, n::Int) = (u = codeunits(s); (String(u[1:n]), String(u[n+1:end])))
const PARAM_END = "\n</parameter>"
json_fragment(x::AbstractString) = chop(JSON3.write(String(x)); head = 1, tail = 1)      # the inside of a JSON string, for a piece of a longer one

# One step of the tool-call state machine on p.buf (see OutParser.tstate). True when it made progress (call again), false when it needs more text.
function _feed_toolcall!(p::OutParser, ev, final::Bool)
    args(t) = (push!(ev, (:tool_args, t)); p.nargs += ncodeunits(t); nothing)
    function close_call!()
        p.topen && (args("}"); push!(ev, (:tool_end, nothing)); p.topen = false)
        p.tstate = :head; p.tname = ""; p.tfirst = true; p.nargs = 0
    end
    if p.tstate == :head
        m = match(r"^\s*<function=([^>\n]+)>"s, p.buf)
        i = findfirst("</tool_call>", p.buf)
        if m === nothing
            if i !== nothing                                         # a tool call without a function: hand it back as content
                push!(ev, (:content, "<tool_call>" * p.buf[1:last(i)])); p.buf = p.buf[last(i)+1:end]; p.mode = :content; p.lead = true
                return true
            end
            if final && !isempty(strip(p.buf))
                push!(ev, (:content, "<tool_call>" * p.buf)); p.buf = ""
            end
            return false
        end
        p.tname = String(m[1]); p.ncalls += 1; p.topen = true; p.nargs = 0
        push!(ev, (:tool_start, p.tname)); args("{")
        p.buf = p.buf[m.match.ncodeunits+1:end]; p.tstate = :between
        return true
    elseif p.tstate == :between
        b = lstrip(p.buf)
        if final && isempty(b) && p.topen                           # the text ended inside the call: close it
            close_call!(); p.mode = :content; p.buf = ""
            return false
        end
        if startswith(b, "<parameter=")
            m = match(r"^<parameter=([^>\n]+)>\n?"s, b)
            (m === nothing || (m.match[end] == '>' && !final)) && return false      # (the tag, or the line break that may follow it, is not complete yet)
            p.tparam = String(m[1])
            str = get(get(p.schema, p.tname, Dict{String,String}()), p.tparam, "") == "string"
            args((p.tfirst ? "" : ",") * JSON3.write(p.tparam) * ":" * (str ? "\"" : ""))
            p.tfirst = false
            p.buf = b[m.match.ncodeunits+1:end]; p.tstate = str ? :value_str : :value_hold
            return true
        elseif startswith(b, "</function>")
            p.buf = b[ncodeunits("</function>")+1:end]
            args("}"); push!(ev, (:tool_end, nothing)); p.topen = false
            return true
        elseif startswith(b, "</tool_call>")
            p.buf = b[ncodeunits("</tool_call>")+1:end]
            close_call!(); p.mode = :content; p.lead = true
            return true
        elseif isempty(b) || (!final && any(t -> startswith(t, b), ("<parameter=", "</function>", "</tool_call>")))
            return false
        elseif startswith(b, "<")                                    # (unknown tag, or the text ended inside one)
            if final
                close_call!(); p.mode = :content; p.buf = ""
                return false
            end
            p.buf = b[nextind(b, 1):end]; return true
        else
            p.buf = b[nextind(b, 1):end]; return true                # stray text between parameters: dropped
        end
    else                                                             # a parameter's value
        i = findfirst(PARAM_END, p.buf)
        if p.tstate == :value_str
            if i !== nothing
                v = p.buf[1:first(i)-1]; isempty(v) || args(json_fragment(v)); args("\"")
                p.buf = p.buf[last(i)+1:end]; p.tstate = :between
                return true
            end
            k = final ? 0 : partial_suffix(p.buf, PARAM_END)
            v, p.buf = split_bytes(p.buf, ncodeunits(p.buf) - k); isempty(v) || args(json_fragment(v))
            if final
                args("\""); close_call!(); p.mode = :content; p.buf = ""
            end
            return false
        else
            if i !== nothing
                v = p.buf[1:first(i)-1]
                val = v
                try; val = JSON3.read(v); catch; end
                args(JSON3.write(val isa AbstractString ? String(val) : val))
                p.buf = p.buf[last(i)+1:end]; p.tstate = :between
                return true
            end
            if final
                args(JSON3.write(p.buf)); close_call!(); p.mode = :content; p.buf = ""
            end
            return false
        end
    end
end

# -> a vector of (:reasoning | :content, text) and, for a tool call, (:tool_start, name), then (:tool_args, fragment of the arguments JSON) as the
# parameters are read (a string value streams character by character; any other type is sent once it is complete), then (:tool_end, nothing).
# The fragments add up to what `tool_arguments` returns for the whole call.
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
            r, p.buf = split_bytes(p.buf, ncodeunits(p.buf) - k); isempty(r) || push!(ev, (:reasoning, r))
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
            head = split_bytes(p.buf, ncodeunits(p.buf) - k)[1]
            c = final ? head : rstrip(head)                       # white space before a (possible) tag stays held back
            isempty(c) || push!(ev, (:content, String(c)))
            p.buf = p.buf[ncodeunits(c)+1:end]
            break
        else
            _feed_toolcall!(p, ev, final) || break
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
    has_system::Bool
    parts::Dict{Symbol,String}
    stop::Vector{String}
    events::Channel{Any}
    cancel::Base.RefValue{Bool}
    budget::Int                  # reasoning tokens allowed before the thinking is cut off (0: no limit)
end
const REASONING_BUDGET = Ref(0)                 # the server's default (--reasoning-budget=N); a request's thinking_budget overrides it
const BUDGET_WRAPUP = "\n\nConsidering the limited thinking budget, I have to give the answer based on the thinking so far now.\n</think>\n\n"

const PREFIX = PrefixCache()                    # the saved prefill states (see `PrefixCache`)
const JOBS = Channel{Job}(64)
const LAST_FULL = Ref(Int[])                    # the previous request's prompt + generated tokens (to see how much of it the next prompt repeats)
const LAST_NP = Ref(0)                          # ... and the length of that prompt

# The part of `text` after byte `sent`, cut by bytes (a decoded token list may hold a lone byte of an unfinished character, which is not valid
# UTF-8: indexing by characters would fail), with any invalid byte shown as U+FFFD.
tail_text(text::String, sent::Int) = (t = String(codeunits(text)[sent+1:end]); isvalid(t) ? t : join(c isa Char && isvalid(c) ? c : '\ufffd' for c in t))

function run_job!(s, d, tok, job::Job, use_spec::Bool)
    te = time(); ids = Int.(LT.encode(tok, job.prompt; bos = false)); t_enc = time() - te
    try                                          # (a diagnostic: could the end-of-generation state of the previous request be reused?)
        lf = LAST_FULL[]; l = Luminal.Decoding._lcp(lf, ids); np = LAST_NP[]
        if !isempty(lf)
            thi = only(Int.(LT.encode(tok, "</think>"; bos = false))); ti = findfirst(==(thi), view(lf, np+1:length(lf)))
            logline(@sprintf("  previous sequence: %d prompt + %d generated tokens (</think> at +%s); this prompt repeats %d of them (%d of the generated)%s",
                             np, length(lf) - np, ti === nothing ? "none" : string(ti), l, max(l - np, 0),
                             l < length(lf) ? "; first difference: previous " * repr(LT.decode(tok, lf[max(l-1,1):min(l+4,end)])) * " vs now " * repr(LT.decode(tok, ids[max(l-1,1):min(l+4,length(ids))])) : " (all)"))
        end
    catch e
        logline("  previous-sequence diagnostic failed: " * sprint(showerror, e))
    end
    t0 = time()
    logline(@sprintf("request: %d prompt tokens, up to %d new", length(ids), job.max_tokens))
    try
        names = ((:tools, "system prompt tools"), (:system, "system text"), (:user, "user"), (:reasoning, "earlier reasoning"), (:assistant, "earlier replies"),
                 (:toolcalls, "tool calls"), (:toolresults, "tool results"), (:markup, "chat markup"))
        cnt = [(label, length(LT.encode(tok, job.parts[k]; bos = false))) for (k, label) in names if haskey(job.parts, k) && !isempty(job.parts[k])]
        logline("  breakdown: " * join(("$label $c" for (label, c) in cnt), ", ") * " (parts sum to $(sum(last, cnt)))")
    catch e
        logline("  breakdown unavailable: " * sprint(showerror, e))
    end
    if length(ids) + job.max_tokens + 8 > s.max_seq
        put!(job.events, (:error, "the prompt has $(length(ids)) tokens and max_tokens is $(job.max_tokens); the server's context is $(s.max_seq)"))
        return
    end
    parser = OutParser(job.thinking, job.tools)
    gen = Int[]; sent = 0                        # bytes of the decoded text already fed to the parser
    reason = :stop; visible = ""
    eos = Set(tok.eos_ids)
    im_start = only(Int.(LT.encode(tok, "<|im_start|>"; bos = false)))
    starts = findall(==(im_start), ids)                            # turn headers: everything before the last one is the same in the next prompt of this
    prefix_at = Pair{Symbol,Int}[]                                 # conversation, and before the second one (a system message ends there) in any with this system prompt
    job.has_system && length(starts) >= 2 && push!(prefix_at, :sys => starts[2] - 1)
    isempty(starts) || push!(prefix_at, :turn => starts[end] - 1)
    stats = Dict{Symbol,Any}()
    budget_hit = Ref(false)
    t_first = Ref(0.0)
    last_ev = Ref(time()); last_log = Ref(time()); last_done = Ref(-1)
    gen_log = Ref(0.0); gen_log_n = Ref(0); nreason = Ref(0); ncontent = Ref(0)      # the generation progress line (every 30 s)
    # during a long prefill: an event for the HTTP side every few seconds (it turns them into keep-alive chunks, so the client's idle
    # timeout does not fire) and a progress line in the log every 30 s
    function on_progress(done, total)
        now = time()
        if now - last_ev[] >= 5
            put!(job.events, (:progress, (done, total))); last_ev[] = now
        end
        last_done[] < 0 && (last_done[] = done)                 # (the first chunk only sets the base: a resumed prompt starts at its cached length)
        if now - last_log[] >= 30
            logline(@sprintf("  prefill %d/%d tokens (%.0f%%), %.0f tok/s over the last %.0f s", done, total, 100 * done / total,
                             (done - last_done[]) / max(now - last_log[], 1e-6), now - last_log[]))
            last_log[] = now; last_done[] = done
        end
        return nothing
    end
    function emit(text; final=false)
        for e in feed!(parser, text; final)
            put!(job.events, e)
            e[1] == :content && (visible *= e[2])
            e[1] == :reasoning && (nreason[] += length(e[2]))
            e[1] == :content && (ncontent[] += length(e[2]))
        end
    end
    gc_t = Ref(time())
    function on_token(new)
        if time() - gc_t[] > 2.0                         # the only place besides the idle loop where the server collects garbage (see `Luminal.hostgc`)
            Luminal.hostgc(false); gc_t[] = time()
        end
        for t in new
            if t in eos; reason = :stop; return true; end
            push!(gen, t)
            t_first[] == 0.0 && (t_first[] = time())
            if length(gen) >= job.max_tokens; reason = :length; break; end
        end
        job.cancel[] && (reason = :cancel; return true)
        if time() - last_ev[] >= 8                      # a long tool call is held back until it is complete: keep the stream alive meanwhile
            put!(job.events, (:progress, (length(gen), 0))); last_ev[] = time()
        end
        if t_first[] > 0 && time() - max(gen_log[], t_first[]) >= 30      # what the model is doing, for the log: a tool call is not streamed until complete
            now = time(); dt = now - max(gen_log[], t_first[])
            held = parser.mode == :toolcall ? @sprintf(", tool call being written (%s: %d chars of arguments sent, %d held back)", isempty(parser.tname) ? "?" : parser.tname,
                                                       parser.nargs, ncodeunits(parser.buf)) : ""
            logline(@sprintf("  generating: %d tokens, %.1f tok/s over the last %.0f s | %s mode | %d chars of reasoning, %d of answer%s",
                             length(gen), (length(gen) - gen_log_n[]) / dt, dt, parser.mode, nreason[], ncontent[], held))
            gen_log[] = now; gen_log_n[] = length(gen)
        end
        text = LT.decode(tok, gen)
        (endswith(text, '\ufffd') || !isvalid(text)) && reason != :length && return false     # an incomplete UTF-8 sequence: wait for its rest
        emit(tail_text(text, sent)); sent = ncodeunits(text)
        if !isempty(job.stop) && any(st -> occursin(st, visible), job.stop)
            reason = :stop; return true
        end
        if job.budget > 0 && !budget_hit[] && parser.mode == :reasoning && length(gen) >= job.budget      # still thinking at the budget: stop here, close the thinking, resume
            budget_hit[] = true; return true
        end
        return reason == :length
    end
    function generate!(prompt, max_new, st, at)
        if d isa DFlash2
            generate_ids_dflash(s, d, prompt; max_new_tokens = max_new + 8, sampler = job.sampler, on_token, prefix = PREFIX, prefix_at = at, stats = st, on_progress)
        elseif use_spec
            generate_ids_speculative(s, d, prompt; max_new_tokens = max_new + 8, ndraft = 0, sampler = job.sampler, on_token, prefix = PREFIX, prefix_at = at, stats = st, on_progress)
        else
            generate_ids(s, [prompt]; max_new_tokens = max_new + 8, sampler = job.sampler, on_token, prefix = PREFIX, prefix_at = at, on_progress)
        end
    end
    try
        generate!(ids, job.max_tokens, stats, prefix_at)
        if budget_hit[] && reason == :stop && !job.cancel[]
            spec1 = haskey(stats, :cycles) && stats[:cycles] > 0 ? spec_summary(stats) : ""
            isempty(spec1) || logline("  speculation (reasoning): " * spec1)
            logline(@sprintf("  reasoning budget reached (%d tokens, %.1f s): closing the thinking, answering", length(gen), time() - t0))
            wrap = Int.(LT.encode(tok, BUDGET_WRAPUP; bos = false))
            append!(gen, wrap)
            text = LT.decode(tok, gen)
            emit(tail_text(text, sent)); sent = ncodeunits(text)
            cached1 = get(stats, :prefill_reused, 0)
            stats = Dict{Symbol,Any}()
            generate!(vcat(ids, gen), max(job.max_tokens - length(gen), 1), stats, Pair{Symbol,Int}[])
            stats[:prefill_reused] = cached1
        end
        text = LT.decode(tok, gen)
        emit(tail_text(text, sent); final = true)
        length(gen) >= job.max_tokens && reason == :stop && !isempty(gen) && !(gen[end] in eos) && (reason = :length)
        t1 = time()
        cached = get(stats, :prefill_reused, 0); tf = t_first[] == 0.0 ? t1 : t_first[]
        spec = haskey(stats, :cycles) && stats[:cycles] > 0 ? spec_summary(stats) : ""
        isempty(spec) || logline("  speculation: " * spec)
        logline("  prefix cache: " * prefix_summary(PREFIX))
        LAST_FULL[] = vcat(ids, gen); LAST_NP[] = length(ids)
        reason == :cancel && logline(@sprintf("  cancelled: the client closed the connection after %d generated tokens (%.1f s)", length(gen), t1 - t0))
        put!(job.events, (:done, (finish = parser.ncalls > 0 && reason == :stop ? "tool_calls" : reason == :length ? "length" : "stop",
                                  prompt_tokens = length(ids), completion_tokens = length(gen), cached_tokens = cached,
                                  prefill_s = tf - t0, total_s = t1 - t0,
                                  phases = @sprintf("resumed from %s, tokenize %.2f s, restore %.2f, chunks %.2f, save %.2f, register %.2f, mtp restore %.2f, mtp prime+save %.2f", PREFIX.resumed_kind, t_enc,
                                                    (get(Luminal.Decoding.PREFILL_TIMES, k, 0.0) for k in (:restore, :chunks, :save, :register, :mtp_restore, :mtp_prime_save))...),
                                  timings = (cache_n = cached, prompt_n = length(ids) - cached, prompt_ms = 1000 * (tf - t0),
                                             prompt_per_second = (length(ids) - cached) / max(tf - t0, 1e-6),
                                             predicted_n = length(gen), predicted_ms = 1000 * (t1 - tf),
                                             predicted_per_second = max(length(gen) - 1, 0) / max(t1 - tf, 1e-6)))))
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

# One line on the MTP speculation of a request: tokens per cycle, how often drafts 1, 2, 3 were accepted (given the earlier ones were), the draft
# counts the controller chose, and where the cycle time went (verify / drafting / rollback).
function spec_summary(st)
    c = st[:cycles]; acc = st[:acc_by_depth]
    rate(j) = j == 1 ? acc[1] / c : acc[j - 1] > 0 ? acc[j] / acc[j - 1] : 0.0
    nd = get(st, :cycles_nd, Int[])
    ms(k) = get(st, k, 0)
    tot = max(ms(:ms_verify) + ms(:ms_mtp) + ms(:ms_rollback), 1)
    @sprintf("%d cycles, %.2f tokens/cycle | accepted: %s | drafts per cycle: %s | verify %.0f ms/cycle, draft %.0f, rollback %.0f",
             c, st[:emitted] / c, join((@sprintf("d%d %.0f%%", j, 100 * rate(j)) for j in eachindex(acc)), ", "),
             join((@sprintf("%d:%.0f%%", j, 100 * nd[j] / c) for j in eachindex(nd) if nd[j] > 0), " "),
             ms(:ms_verify) / c, ms(:ms_mtp) / c, ms(:ms_rollback) / c)
end

function log_timings(val)
    t = val.timings
    logline(@sprintf("prompt processing: %d new tokens (+%d cached) in %.2f s = %.0f tok/s | generation: %d tokens in %.2f s = %.1f tok/s | %s",
                     t.prompt_n, t.cache_n, t.prompt_ms / 1000, t.prompt_per_second, t.predicted_n, t.predicted_ms / 1000, t.predicted_per_second, val.finish))
    haskey(val, :phases) && logline("  time to first token: " * val.phases)
end

function chat_handler(http::HTTP.Stream, name::String)
    req = JSON3.read(String(read(http)))
    messages = field(req, :messages)
    messages === nothing && return err_response(http, 400, "messages is required")
    tools = field(req, :tools)
    kwargs = field(req, :chat_template_kwargs, Dict())
    thinking = field(kwargs, :enable_thinking, true) !== false
    effort = String(something(field(req, :reasoning_effort), field(kwargs, :reasoning_effort), "medium"))
    parts = Dict{Symbol,IOBuffer}()
    prompt = try
        render_chat(messages, tools; thinking, effort, parts)
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
    # the reasoning budget: thinking_budget (Qwen's API), thinking.budget_tokens (Anthropic's), reasoning_budget, or the same in chat_template_kwargs
    budget = something(field(req, :thinking_budget), (th = field(req, :thinking); th isa AbstractDict || th isa JSON3.Object ? field(th, :budget_tokens) : nothing), field(req, :reasoning_budget),
                       field(kwargs, :thinking_budget), field(kwargs, :reasoning_budget), REASONING_BUDGET[])
    budget = budget isa Number ? max(Int(budget), 0) : REASONING_BUDGET[]
    logline("  request fields: " * join(string.(keys(req)), ", ") * (budget > 0 ? "; reasoning budget $budget tokens" : ""))
    job = Job(prompt, max_tokens, sampler, tools, thinking, String(field(messages[1], :role, "")) in ("system", "developer"), Dict(k => String(take!(v)) for (k, v) in parts), stop, Channel{Any}(Inf), Ref(false), thinking ? budget : 0)
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
                elseif kind == :progress
                    write(http, val[2] == 0 ? ": generating, $(val[1]) tokens so far" : ": prefill $(val[1])/$(val[2]) tokens", "\n\n")   # an SSE comment: no "data:" prefix
                    sse(chunk((;)))                             # an empty delta ({}): a chunk any client counts as activity, and changes no client state
                                                                # (an empty `content` string may end the client's open reasoning block: later thinking was not shown)
                elseif kind == :tool_start
                    sse(chunk((tool_calls = [(index = ncall, id = "call_" * randstring(12), type = "function",
                                              var"function" = (name = val, arguments = ""))],)))
                elseif kind == :tool_args
                    sse(chunk((tool_calls = [(index = ncall, var"function" = (arguments = val,))],)))
                elseif kind == :tool_end
                    ncall += 1
                elseif kind == :done
                    u = (prompt_tokens = val.prompt_tokens, completion_tokens = val.completion_tokens, total_tokens = val.prompt_tokens + val.completion_tokens,
                         prompt_tokens_details = (cached_tokens = val.cached_tokens,))
                    sse(merge(chunk((;); finish = val.finish, usage = u), (timings = val.timings,)))
                    sse("[DONE]")
                    log_timings(val)
                    break
                else
                    sse((error = (message = val, type = "server_error"),)); sse("[DONE]"); break
                end
            end
        catch e
            job.cancel[] = true                                              # the client went away: stop generating
        end
    else
        reasoning = IOBuffer(); content = IOBuffer(); calls = Any[]; fin = nothing; usage = nothing; timings = nothing
        while true
            kind, val = take!(job.events)
            kind == :reasoning && print(reasoning, val)
            kind == :content && print(content, val)
            kind == :tool_start && (push!(calls, (id = "call_" * randstring(12), type = "function", name = val, args = IOBuffer())))
            kind == :tool_args && print(calls[end].args, val)
            if kind == :done
                fin = val.finish; timings = val.timings; log_timings(val); usage = (prompt_tokens = val.prompt_tokens, completion_tokens = val.completion_tokens, total_tokens = val.prompt_tokens + val.completion_tokens,
                                           prompt_tokens_details = (cached_tokens = val.cached_tokens,)); break
            elseif kind == :error
                return err_response(http, 500, val)
            end
        end
        msg = (role = "assistant", content = String(take!(content)), reasoning_content = String(take!(reasoning)))
        isempty(calls) || (msg = merge(msg, (tool_calls = [(id = c.id, type = c.type, var"function" = (name = c.name, arguments = String(take!(c.args)))) for c in calls],)))
        json_response(http, 200, (id = id, object = "chat.completion", created = created, model = name,
                                  choices = [(index = 0, message = msg, finish_reason = fin)], usage = usage, timings = timings))
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
        e isa Base.IOError || e isa EOFError || e isa HTTP.Exceptions.HTTPError || @error "request failed" exception = (e, catch_backtrace())
        try; err_response(http, 500, sprint(showerror, e)); catch; end
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------

# Run `f` in a task pinned to a thread other than the main one (thread 1), whose tasks serve the HTTP clients: a blocking GPU wait on that
# thread would starve them. (A spawned task may start on thread 1, and may migrate later: pin it.)
function spawn_off_main_thread(f)
    while true
        started = Channel{Bool}(1)
        t = Threads.@spawn begin
            tid = Threads.threadid()
            ok = tid != 1 && ccall(:jl_set_task_tid, Cint, (Any, Cint), current_task(), tid - 1) == 1
            ok && (current_task().sticky = true)
            put!(started, ok)
            ok && f()
        end
        take!(started) && return t
        wait(t)
    end
end

function main()
    Logging.global_logger(QuietHTTP(Logging.global_logger()))
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(n, d) = (i = findfirst(a -> startswith(a, "--$n="), ARGS); i === nothing ? d : ARGS[i][length(n)+4:end])
    isempty(args) && error("usage: qwen35_server.jl <model.gguf> [--mtp=<gguf>] [--port=8080] [--host=127.0.0.1] [--max-seq=16384] [--int8] [--kv-int8] [--no-spec] [--dflash=<gguf>] [--name=id] [--reasoning-budget=N]")
    base = args[1]; mtp = flag("mtp", base)
    port = parse(Int, flag("port", "8080")); host = flag("host", "127.0.0.1")
    use_spec = !("--no-spec" in ARGS)
    dflash_path = flag("dflash", "")                     # the DFlash2 block drafter instead of the MTP head
    use_dflash = use_spec && !isempty(dflash_path)
    REASONING_BUDGET[] = parse(Int, flag("reasoning-budget", "0"))
    name = flag("name", splitext(basename(base))[1])
    T = "--int8" in ARGS ? Int8 : Luminal.Int4
    KV = "--kv-int8" in ARGS ? Int8 : Float16            # --kv-int8: half the K/V cache (per-token absmax scales), see src/DecodeAttn.jl
    # All GPU work (loading, warm-up, the generation loop) runs in one task off the main thread: the HTTP server's tasks stay on the main
    # thread, and a blocking GPU wait there would starve them (nothing is written to the clients between the end of a prefill and the end of the reply).
    ready = Channel{Any}(1)
    worker = spawn_off_main_thread() do
        try
        f = read_gguf(base); cfg = qwen35_config(f); tok = LlamaTokenizer(f)
        logline("loading $base")
        W = load_qwen35_gguf(base, cfg; quantize = _ -> T)
        use_spec && !use_dflash && merge!(W, load_qwen35_mtp(mtp, cfg; quantize = _ -> T, block = cfg.n_layers))
        s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = parse(Int, flag("max-seq", "16384")), weights = W, decode_weights = T,
                         kv_dtype = KV)
        d = use_dflash ? DFlash2(s, dflash_path, base) : use_spec ? MTPDrafter(s) : nothing
        logline("loaded; warming up (compiles the graphs, about a minute)")
        gen_spec(prompt; kw...) = d isa DFlash2 ? generate_ids_dflash(s, d, prompt; kw...) : generate_ids_speculative(s, d, prompt; ndraft = 0, kw...)
        wid = Int.(LT.encode(tok, render_chat([(role = "user", content = "Say hello.")], nothing); bos = false))
        for sm in (nothing, Sampler(temperature = 1.0, top_p = 0.95, top_k = 20))
            use_spec ? (gen_spec(wid; max_new_tokens = 24, sampler = sm); gen_spec(wid; max_new_tokens = 24, sampler = sm)) :
                       generate_ids(s, [wid]; max_new_tokens = 8, sampler = sm)
        end
        if d isa DFlash2                               # one verify graph per block length (2..8 rows)
            for nd in 1:7
                generate_ids_dflash(s, d, wid; max_new_tokens = 6, nprop = nd)
            end
        end
        if use_spec                                    # compile the short-chunk prefill graphs (64 and 256 tokens) and the 512 one, through a throwaway cache
            scratch = PrefixCache(min_tokens = typemax(Int))
            for len in (20, 150, 700)
                gen_spec(repeat(wid, cld(len, length(wid)))[1:len]; max_new_tokens = 4, prefix = scratch)
            end
        end
        Luminal.hostgc(true)
        GC.enable(false)        # From here on only this thread collects garbage: a finalizer freeing GPU memory on the HTTP thread while this one launches a captured graph
                                # deadlocks inside the HIP runtime (seen with the first request that has tools arriving while the previous one finished).
        put!(ready, s.max_seq)
        while true
            if !isready(JOBS)                            # idle: collect now and then (the HTTP side allocates a little)
                timedwait(() -> isready(JOBS), 10.0)
                isready(JOBS) || (Luminal.hostgc(true); continue)
            end
            job = take!(JOBS)
            job.cancel[] && continue
            run_job!(s, d, tok, job, use_spec)
            release_stale!(PREFIX)                       # (the replaced prefix entries: freed here, between requests, not before the first token)
            Luminal.reclaim!(s.device)
        end
        catch e
            @error "worker failed" exception = (e, catch_backtrace())
            put!(ready, e); rethrow()
        end
    end
    max_seq = take!(ready); max_seq isa Exception && error("loading failed: " * sprint(showerror, max_seq))
    server = HTTP.serve!(http -> route(http, name), host, port; stream = true)
    logline("serving $name on http://$host:$port/v1 (speculative decoding: $use_spec$(use_dflash ? ", DFlash2 block drafter" : use_spec ? ", MTP head" : ""), context $max_seq, $(KV === Int8 ? "int8" : "Float16") KV cache)")
    wait(worker)
end

(abspath(PROGRAM_FILE) == (@__FILE__)) && main()
