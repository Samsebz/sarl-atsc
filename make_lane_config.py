import argparse
import json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--width",type=int,required=True); parser.add_argument("--height",type=int,required=True); parser.add_argument("--lanes",type=int,default=4); parser.add_argument("--output",default="configs/lane_masks.json"); args=parser.parse_args()
    step=args.width//args.lanes; lanes=[]
    for i in range(args.lanes):
        x1=i*step; x2=args.width if i==args.lanes-1 else (i+1)*step; lanes.append([[x1,0],[x2,0],[x2,args.height],[x1,args.height]])
    with open(args.output,"w") as f: json.dump({"lanes":lanes},f,indent=2)
    print(f"Saved {args.output}")
if __name__=="__main__": main()
