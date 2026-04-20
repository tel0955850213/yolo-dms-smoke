from ultralytics import YOLO

if __name__ == '__main__':
    # 載入已訓練好的模型
    model = YOLO("C:/Users/Lin/Desktop/CLAUDE_CRAZY/yolov10/best.pt")

    results = model.train(
        data="C:/Users/Lin/Desktop/CLAUDE_CRAZY/merged_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.0005,       # fine-tuning 用小 learning rate
        freeze=10,        # 凍結前10層，保留原有知識
        name="finetune_with_smoke",
        project="C:/Users/Lin/Desktop/CLAUDE_CRAZY/runs",
    )

    print(f"完成！模型儲存在: {results.save_dir}")
