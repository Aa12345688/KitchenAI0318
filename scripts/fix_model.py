import sys
import os

def run_fix():
    print("🚀 YOLO DNA 轉譯工具 3.0 (PT/ONNX -> FP32 ONNX)")
    
    # 🚀 優先檢查 ultralytics (可以直接從 .pt 轉 FP32)
    try:
        from ultralytics import YOLO
        has_ultralytics = True
    except ImportError:
        has_ultralytics = False

    try:
        import onnx
        from onnxconverter_common import float16_to_float32
        has_onnx_tools = True
    except ImportError:
        has_onnx_tools = False

    # 🚀 路徑尋找
    input_pt = None
    input_onnx = None
    
    search_dirs = [".", "Fridge-AI-Master-2-main", "public", "Fridge-AI-Master-2-main/public"]
    for d in search_dirs:
        pt_path = os.path.join(d, "best.pt")
        onnx_path = os.path.join(d, "best.onnx")
        if os.path.exists(pt_path) and not input_pt:
            input_pt = pt_path
        if os.path.exists(onnx_path) and not input_onnx:
            input_onnx = onnx_path

    # 目標路徑 (統一輸出到專案的 public 資料夾)
    target_dir = "Fridge-AI-Master-2-main/public" if os.path.exists("Fridge-AI-Master-2-main/public") else "public"
    output_path = os.path.join(target_dir, "best.onnx")

    # --- 策略 A: 從 .pt 轉檔 ---
    if input_pt and has_ultralytics:
        print(f"📦 偵測到 PyTorch 模型：{input_pt}")
        print("⚡ 正在執行 ultralytics FP32 匯出...")
        model = YOLO(input_pt)
        model.export(format="onnx", half=False, imgsz=640)
        # ultralytics 通常會匯出在同目錄，我們移動它
        exported_path = input_pt.replace(".pt", ".onnx")
        if os.path.exists(exported_path):
            import shutil
            shutil.move(exported_path, output_path)
            print(f"✅ 成功從 .pt 匯出並移動至：{output_path}")
        return

    # --- 策略 B: 從 ONNX 轉檔 ---
    if input_onnx and has_onnx_tools:
        print(f"📦 偵測到 ONNX 模型：{input_onnx}")
        print("⚡ 正在執行 FLOAT16 -> FP32 轉譯...")
        model = onnx.load(input_onnx)
        model_fp32 = float16_to_float32(model)
        onnx.save(model_fp32, output_path)
        print(f"✅ 成功完成轉譯：{output_path}")
        return

    print("❌ 錯誤：無法執行轉換。")
    if not input_pt and not input_onnx:
        print("💡 找不到 'best.pt' 或 'best.onnx'。請確保檔案放在專案目錄下。")
    if not has_ultralytics and not has_onnx_tools:
        print("💡 請安裝必要環境：pip install ultralytics onnx onnxconverter-common")

if __name__ == "__main__":
    run_fix()
