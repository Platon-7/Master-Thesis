import re, zlib, json, sys

def content(pdf):
    d=open(pdf,'rb').read()
    best=b''
    for s in re.findall(rb'stream\r?\n(.*?)endstream', d, re.S):
        try: u=zlib.decompress(s)
        except Exception: continue
        if len(u)>len(best) and b' l' in u: best=u
    return best.decode('latin-1')

def parse(txt):
    toks=txt.replace('\n',' ').replace('[',' [ ').replace(']',' ] ').split()
    nums=[]; cur=[]; start=None
    gs={'RG':(0,0,0),'rg':(0,0,0),'clip':None,'dash':None}
    stack=[]; items=[]
    i=0
    while i<len(toks):
        t=toks[i]
        if re.fullmatch(r'-?\d*\.?\d+', t): nums.append(float(t)); i+=1; continue
        if t=='[':                      # dash array
            j=i
            while j<len(toks) and toks[j]!=']': j+=1
            gs['dash']=' '.join(toks[i:j+1]); nums=[]; i=j+1; continue
        if t=='q': stack.append(dict(gs))
        elif t=='Q':
            if stack: gs=stack.pop()
        elif t=='RG' and len(nums)>=3: gs['RG']=tuple(round(x,4) for x in nums[-3:])
        elif t=='G'  and len(nums)>=1: gs['RG']=tuple([round(nums[-1],4)]*3)
        elif t=='rg' and len(nums)>=3: gs['rg']=tuple(round(x,4) for x in nums[-3:])
        elif t=='re' and len(nums)>=4: gs['_re']=tuple(nums[-4:])
        elif t=='W': gs['clip']=gs.get('_re')
        elif t=='m' and len(nums)>=2: cur=[(nums[-2],nums[-1])]
        elif t=='l' and len(nums)>=2: cur.append((nums[-2],nums[-1]))
        elif t=='c' and len(nums)>=6: cur.append((nums[-2],nums[-1]))
        elif t in ('S','f','B','b','f*'):
            if len(cur)>=2:
                items.append({'op':t,'stroke':gs['RG'],'fill':gs['rg'],
                              'clip':gs.get('clip'),'dash':gs.get('dash'),
                              'pts':list(cur)})
            cur=[]
        elif t=='n': cur=[]
        nums=[]; i+=1
    return items

def texts(txt):
    out=[]
    for cm,b in re.findall(r'([-\d. ]*cm)?\s*BT(.*?)ET', txt, re.S):
        lit=''.join(re.findall(r'\(((?:[^()\\]|\\.)*)\)', b))
        m=re.findall(r'(-?[\d.]+)', cm or '')
        if lit.strip() and len(m)>=6: out.append((float(m[4]),float(m[5]),lit))
    return out

if __name__=='__main__':
    pdf=sys.argv[1]
    c=content(pdf); open(pdf+'.txt','w').write(c)
    items=parse(c); tx=texts(c)
    json.dump({'items':items,'texts':tx}, open(pdf+'.json','w'))
    print(pdf, '-> items:', len(items), 'texts:', len(tx))
    from collections import Counter
    print(' painted op counts:', Counter(x['op'] for x in items))
    print(' subpath sizes:', sorted(Counter(len(x['pts']) for x in items).items()))
