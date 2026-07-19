"""
Service: Face Recognition (nhan dien khuon mat + gan nhan nguoi).

Kien truc:
  __data/<ten_nguoi>/*.jpg   du lieu train da gan nhan
  __data/_inbox/*.jpg        anh tho ban nhet vao cho di gan nhan
  __data/db.pkl              embeddings sau khi train (tu sinh)
  models/*.onnx              YuNet + SFace (tu tai lan dau)

4 che do:
  enroll   Thu mat tu webcam vao __data/<ten>/   (nhanh nhat de tao du lieu)
  label    Mo man hinh gan nhan cho anh trong __data/_inbox/
  train    Tao db.pkl tu du lieu da gan nhan
  camera   Chay camera nhan dien va dan nhan ten len tung nguoi

Toi uu Intel Iris Xe qua OpenCV DNN target OpenCL (khong can CUDA); tu lui ve CPU.

Vi du:
  python main.py enroll --name "Nguyen Van A"
  python main.py label
  python main.py train
  python main.py camera
  python main.py camera --device cpu --source 0
"""
import argparse
import sys

# Ep stdout/stderr sang UTF-8 de in ten co dau (tieng Viet) khong crash tren
# console codepage mac dinh cua Windows (cp1252/cp437) -> tranh tat cua so.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa
        pass


def main():
    ap = argparse.ArgumentParser(description="Face Recognition service")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto",
                        choices=["auto", "opencl_fp16", "opencl", "cpu"],
                        help="Backend OpenCV DNN (auto: thu OpenCL truoc, loi thi CPU)")

    p_enroll = sub.add_parser("enroll", parents=[common], help="Thu mat tu webcam")
    p_enroll.add_argument("--name", required=True, help="Ten nguoi (co the co dau)")
    p_enroll.add_argument("--source", default="0", help="0=webcam hoac video")
    p_enroll.add_argument("--target", type=int, default=20, help="So anh muc tieu")
    p_enroll.add_argument("--interval", type=float, default=0.4,
                          help="Giay giua 2 lan khi tu dong chup")

    sub.add_parser("label", parents=[common], help="Man hinh gan nhan anh trong _inbox")

    sub.add_parser("train", parents=[common], help="Tao db.pkl tu du lieu da gan nhan")

    p_cam = sub.add_parser("camera", parents=[common], help="Camera nhan dien truc tiep")
    p_cam.add_argument("--source", default="0", help="0=webcam hoac video")
    p_cam.add_argument("--threshold", type=float, default=None,
                       help="Nguong cosine (mac dinh 0.363)")
    p_cam.add_argument("--det-size", type=int, default=320,
                       help="Kich thuoc dau vao detector (nho hon = nhanh hon)")

    args = ap.parse_args()

    if args.cmd == "enroll":
        import enroll
        enroll.run(args.name, args.source, args.device, args.target, args.interval)
    elif args.cmd == "label":
        import label_gui
        label_gui.run()
    elif args.cmd == "train":
        import trainer
        trainer.build(args.device)
    elif args.cmd == "camera":
        import camera
        from recognizer import COSINE_THRESHOLD
        th = args.threshold if args.threshold is not None else COSINE_THRESHOLD
        camera.run(args.source, args.device, th, args.det_size)


if __name__ == "__main__":
    main()
