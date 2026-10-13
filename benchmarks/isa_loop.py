import re,sys,collections
def analyze(path,lo=None,hi=None,name=""):
    txt=open(path).read().split("\n")
    if lo: txt=txt[lo-1:hi]
    labels={}
    for i,l in enumerate(txt):
        m=re.match(r'^(\.LBB\d+_\d+):',l)
        if m: labels[m.group(1)]=i
    best=None
    for i,l in enumerate(txt):
        m=re.search(r'(s_cbranch_\w+|s_branch)\s+(\.LBB\d+_\d+)',l)
        if m and m.group(2) in labels and labels[m.group(2)]<i:
            span=(labels[m.group(2)],i)
            if best is None or (span[1]-span[0])>(best[1]-best[0]): best=span
    a,b=best
    body=[x.strip() for x in txt[a:b+1]]
    ins=[x for x in body if x and not x.startswith(('.',';','//')) and not x.endswith(':')]
    c=collections.Counter(x.split()[0] for x in ins)
    cat=collections.Counter()
    for k,v in c.items():
        if k.startswith('v_wmma'): cat['wmma']+=v
        elif k.startswith('ds_load') or k.startswith('ds_read'): cat['ds_load']+=v
        elif k.startswith('ds_store') or k.startswith('ds_write'): cat['ds_store']+=v
        elif k.startswith('global_load') or k.startswith('buffer_load') or k.startswith('flat_load'): cat['vmem_load']+=v
        elif k.startswith('global_store'): cat['vmem_store']+=v
        elif k=='s_waitcnt' or k.startswith('s_waitcnt'): cat['waitcnt']+=v
        elif k=='s_barrier' or k.startswith('s_barrier'): cat['barrier']+=v
        elif k.startswith('v_'): cat['valu']+=v
        elif k.startswith('s_delay'): cat['s_delay_alu']+=v
        elif k.startswith('s_'): cat['salu/other']+=v
        else: cat[k]+=v
    print("== %s: loop lines %d..%d  instructions %d"%(name,a,b,len(ins)))
    for k,v in sorted(cat.items(),key=lambda x:-x[1]): print("   %-12s %d"%(k,v))
    print("   top ops:", ", ".join("%s %d"%(k,v) for k,v in c.most_common(12)))
    return ins
a=analyze("gdr/fp16.s",1175,2175,"gufo HalfPrefillGemm 256x256 Q4_K Full")
b=analyze("g.s",None,None,"ours _wmma_q4_g_kernel!")
