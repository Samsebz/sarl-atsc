import argparse
from pathlib import Path
import cv2
from sarl_atsc.pipeline import SARLATSC

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",required=True); parser.add_argument("--weights",default="yolov8n.pt"); parser.add_argument("--model",required=True); parser.add_argument("--lane-config",default="configs/lane_masks.example.json"); parser.add_argument("--output",default="artifacts/result.mp4"); args=parser.parse_args()
    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened(): raise RuntimeError(f"Could not open source: {args.source}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height)); system=SARLATSC(lane_config=args.lane_config,model_path=args.model,yolo_weights=args.weights)
    while True:
        ok,frame=cap.read()
        if not ok: break
        annotated,info=system.process_frame(frame,fps=fps)
        text=f"Phase: {info['phase_name']} | Proposed: {info['proposed_action']} | Safe: {info['approved_action']} | Override: {info['overridden']}"
        cv2.putText(annotated,text,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2); writer.write(annotated); cv2.imshow("SARL-ATSC",annotated)
        if cv2.waitKey(1)&0xFF==ord("q"): break
    cap.release(); writer.release(); cv2.destroyAllWindows(); print(f"Saved: {args.output}")

if __name__=="__main__": main()
