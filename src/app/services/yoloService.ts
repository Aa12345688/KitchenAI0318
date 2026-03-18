/**
 * YOLO 辨識引擎服務 (YOLO Inference Service)
 * 
 * 負責 ONNX Runtime 初始化與模型載入。
 * 支援「全域預熱 (Pre-warming)」，讓應用程式啟動時即開始加載大體積權重。
 */

export class YOLOService {
    private session: any = null;
    private isInitializing: boolean = false;
    private isReady: boolean = false;
    private lastError: string | null = null;
    private forceWasm: boolean = false; // 新增：強制 WASM 模式 (Safe Mode)

    // 類別名稱對照表
    public readonly CLASS_NAMES = [
        "apple", "banana", "cabbage", "meat", "orange",
        "rotten apple", "rotten banana", "rotten cabbage",
        "rotten meat", "rotten orange", "rotten spinach", "spinach"
    ];

    /**
     * 預熱模型：在 App 啟動時即刻觸發，不需等待用戶進入 CameraView
     */
    public async prewarm() {
        if (this.isInitializing || this.isReady) return;
        this.isInitializing = true;
        this.lastError = null;
        
        console.log("🧠 [YOLO] 核心啟動中 (背景預熱)...");
        try {
            const ort = (window as any).ort;
            if (!ort) {
                console.warn("⚠️ 找不到 ort 引擎，延後初始化");
                this.isInitializing = false;
                return;
            }

            const baseUrl = import.meta.env.BASE_URL || "/";
            const startTime = Date.now();
            
            // 🚀 [Universal Compatibility] 指向本地 FULL BUNDLE 目錄
            ort.env.wasm.wasmPaths = `${baseUrl}wasm/`;
            ort.env.wasm.numThreads = 1; 

            // 支援雙版本模型載入：優先嘗試標準版，失敗則嘗試通用版 (FP32)
            const modelUrls = [
                `${baseUrl}best.onnx?v=1.1.3`,
                `${baseUrl}best_universal.onnx?v=1.1.3`
            ];
            
            // 🚀 [Failover Sequence] 根據 Safe Mode 調整順序
            const providers = this.forceWasm ? ["wasm"] : ["webgpu", "webgl", "wasm"];
            let recentError = "";
            
            for (const modelUrl of modelUrls) {
                if (this.session) break;

                for (const provider of providers) {
                    try {
                        console.log(`🧠 [YOLO] 嘗試建立會話 (模型: ${modelUrl.split('?')[0]}, 後端: ${provider})...`);
                        this.session = await ort.InferenceSession.create(modelUrl, {
                            executionProviders: [provider],
                            graphOptimizationLevel: "all",
                            enableCpuMemArena: true,
                            enableMemPattern: true
                        });
                        console.log(`✅ [YOLO] 成功掛載核心 (使用: ${provider})`);
                        break; 
                    } catch (err: any) {
                        recentError = err.message || "未知錯誤";
                        console.warn(`⚠️ [YOLO] 嘗試失敗: ${recentError}`);
                        
                        if (recentError.includes("float16") && provider === "wasm") {
                            console.warn("💡 [Optimization Tip] 此裝置不支援 FLOAT16 加速。");
                        }
                    }
                }
            }

            if (!this.session) {
                console.error("🏁 [Summary] 所有相容性嘗試皆失敗。請參考 scripts/fix_model.py 將模型轉為 FP32 格式。");
                this.lastError = recentError; 
                throw new Error(`全環境初始化失敗。原因：硬體不支援 FLOAT16。請更換為 FP32 模型。`);
            }

            this.isReady = true;
            this.isInitializing = false;
            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            console.log(`✅ [YOLO] 引擎就緒 (總耗時 ${duration}s)`);
        } catch (e: any) {
            console.error("❌ [YOLO] 預熱失敗:", e);
            this.lastError = e.message || "初始化失敗";
            this.isInitializing = false;
        }
    }

    public getSession() {
        return this.session;
    }

    public isLoaded() {
        return this.isReady;
    }

    public isBusy() {
        return this.isInitializing;
    }

    public getError() {
        return this.lastError;
    }

    public isFailed() {
        return !!this.lastError;
    }

    /**
     * 強制重新載入：清除狀態並重新執行預熱
     */
    public async forceReload(wasmOnly: boolean = false) {
        this.forceWasm = wasmOnly;
        this.isReady = false;
        this.isInitializing = false;
        this.session = null;
        await this.prewarm();
    }

    public isSafeMode() {
        return this.forceWasm;
    }
}

export const yoloService = new YOLOService();
