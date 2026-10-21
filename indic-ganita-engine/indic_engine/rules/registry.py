"""Engine 1: verse-referenced rule registry. Every rule cites text, chapter/verse, pdf page where checked, and says whether it is implemented."""
import json, os
_D=os.path.join(os.path.dirname(__file__),"data")
def _load():
    out=[]
    for f in sorted(os.listdir(_D)):
        if f.endswith(".json"): out+=json.load(open(os.path.join(_D,f),encoding="utf-8"))
    return out
RULES=_load()
def list_sources(): 
    s={}
    for r in RULES: s.setdefault(r["text"],[0,0]); s[r["text"]][0]+=1; s[r["text"]][1]+= r["status"]=="implemented"
    return [{"text":k,"rules":v[0],"implemented":v[1]} for k,v in s.items()]
def get_rule(rid): return next((r for r in RULES if r["id"]==rid),None)
def search_rules(q="",text=None,status=None):
    q=q.lower()
    return [r for r in RULES if (not text or r["text"].lower()==text.lower()) and (not status or r["status"]==status)
            and (not q or q in json.dumps(r,ensure_ascii=False).lower())]
