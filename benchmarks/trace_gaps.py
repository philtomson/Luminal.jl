import csv,sys,re,collections
f=sys.argv[1]; minspan=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
rows=[(int(r['Start_Timestamp']),int(r['End_Timestamp']),re.sub(r'\(.*','',r['Kernel_Name'])) for r in csv.DictReader(open(f))]
rows=[r for r in rows if not r[2].startswith('__amd_rocclr')] or rows
rows.sort()
# split into runs at idle periods > 300 ms
runs=[[rows[0]]]
for r in rows[1:]:
    if r[0]-runs[-1][-1][1] > 300e6: runs.append([r])
    else: runs[-1].append(r)
print("runs:",[(len(x),round((x[-1][1]-x[0][0])/1e9,2)) for x in runs])
run=[x for x in runs if (x[-1][1]-x[0][0])/1e9>=minspan][-1]
# busy = union of intervals (kernels may overlap across streams)
busy=0; end=run[0][0]; gaps=[]
for s,e,n in run:
    if s>end: gaps.append((s-end,n)); busy+=e-s
    else: busy+= max(0,e-end)
    end=max(end,e)
span=run[-1][1]-run[0][0]
sumd=sum(e-s for s,e,_ in run)
print("kernels %d  span %.3f s  busy(union) %.3f s  sum durations %.3f s  idle %.3f s (%.1f%%)"%(len(run),span/1e9,busy/1e9,sumd/1e9,(span-busy)/1e9,100*(span-busy)/span))
b=collections.Counter(); bt=collections.Counter()
for g,_ in gaps:
    k = '<2us' if g<2e3 else '2-10us' if g<10e3 else '10-100us' if g<100e3 else '>100us'
    b[k]+=1; bt[k]+=g
for k in ('<2us','2-10us','10-100us','>100us'): print("  gaps %-9s n=%6d  total %.1f ms"%(k,b[k],bt[k]/1e6))
big=collections.Counter()
for g,n in gaps:
    if g>=10e3: big[n[:50]]+=g
print("kernel after large (>=10us) gaps:")
for n,g in big.most_common(8): print("   %.1f ms  %s"%(g/1e6,n))
# small kernels
sm=collections.defaultdict(lambda:[0,0]);
for s,e,n in run:
    if e-s<20e3: sm[n[:50]][0]+=1; sm[n[:50]][1]+=e-s
print("kernels <20us: %d, total %.1f ms"%(sum(v[0] for v in sm.values()),sum(v[1] for v in sm.values())/1e6))
for n,(c,d) in sorted(sm.items(),key=lambda x:-x[1][1])[:8]: print("   %6d  %.1f ms  %s"%(c,d/1e6,n))
