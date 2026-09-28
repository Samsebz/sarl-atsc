import argparse
from ultralytics import YOLO

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data",required=True,help="YOLO dataset YAML"); parser.add_argument("--model",default="yolov8n.pt"); parser.add_argument("--epochs",type=int,default=50); parser.add_argument("--imgsz",type=int,default=640); parser.add_argument("--batch",type=int,default=16); args=parser.parse_args()
    model=YOLO(args.model); model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,project="artifacts/yolo",name="sarl_atsc_detector")
if __name__=="__main__": main()
