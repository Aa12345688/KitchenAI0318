/**
 * LLM 食譜生成服務 (LLM Recipe Generation Service - Direct Gemini API)
 * 
 * 這個模組負責與 Google Gemini API 進行通訊，
 * 根據用戶提供的食材清單動態生成食譜建議。
 * 支援多模型輪詢與多金鑰自動切換。
 */

export interface LLMRecipeRequest {
    ingredients: string[]; // 用戶擁有的食材
    preferences?: string; // 用戶偏好 (可選)
}

export interface LLMRecipe {
    id: string;
    name: string;
    image: string;
    time: string;
    difficulty: "easy" | "medium" | "hard";
    category: "vegetable" | "fruit" | "meat" | "mixed";
    requiredIngredients: string[];
    description: string;
    matchScore: number;
    steps?: { title: string; description: string }[];
    sustainabilityTip?: string; // 新增：環保減廢小撇步
    substitutionTip?: string;  // 新增：食材替代建議
}

// 支援的模型列表，按優先順序排列 (優先使用最新且穩定的版本)
const GEMINI_MODELS = [
    "gemini-3.1-flash-lite", 
    "gemini-3.1-flash",
    "gemini-3-flash",
    "gemini-2.5-flash", 
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro"
];

class LLMService {
    private currentModelIndex = 0;
    private apiKeyIndex = 0;
    private preferredModel: string | null = null;
    private customApiKeys: string[] = [];

    // 記錄每個 key 的冷卻時間 (429 後暫停)
    private keyCooldowns: Record<string, number> = {};
    // 記錄模型是否失效 (404 後略過)
    private modelAvailability: Record<string, boolean> = {};

    private readonly MAX_RETRIES = 2;
    private readonly COOLDOWN_MS = 60000; // 延長冷卻時間至 60s

    setPreferredModel(model: string | null) {
        this.preferredModel = model;
        if (model) {
            const idx = GEMINI_MODELS.indexOf(model);
            if (idx !== -1) this.currentModelIndex = idx;
        }
    }

    getPreferredModel(): string | null {
        return this.preferredModel;
    }

    setCustomApiKeys(keys: string | string[]) {
        if (Array.isArray(keys)) {
            this.customApiKeys = keys.filter(k => k.trim() !== "");
        } else {
            this.customApiKeys = keys.split(",").map(k => k.trim()).filter(k => k !== "");
        }
        // Reset index when keys change
        this.apiKeyIndex = 0;
    }

    private getApiKeys(): string[] {
        // 1. 優先使用用戶在 UI 手動輸入的金鑰 (Prioritize custom keys entered via UI)
        if (this.customApiKeys.length > 0) {
            return this.customApiKeys;
        }

        // 2. 自動偵測環境變數 (Auto-detect environment variables)
        const keys: string[] = [];
        
        // 嘗試讀取逗號隔開的主變數
        const mainKey = import.meta.env.VITE_LLM_API_KEY || "";
        if (mainKey) {
            keys.push(...mainKey.split(",").map((k: string) => k.trim()).filter((k: string) => k !== ""));
        }

        // 嘗試讀取序號形式的變數 (VITE_LLM_API_KEY_1, _2, _3...)
        // 注意：Vite 的 import.meta.env 只能讀取以 VITE_ 開頭且在編譯時已定義的變數
        for (let i = 1; i <= 5; i++) {
            const keyName = `VITE_LLM_API_KEY_${i}`;
            const val = (import.meta.env as any)[keyName];
            if (val && !keys.includes(val.trim())) {
                keys.push(val.trim());
            }
        }

        return keys;
    }

    private isKeyCoolingDown(key: string): boolean {
        const coolUntil = this.keyCooldowns[key];
        return coolUntil ? Date.now() < coolUntil : false;
    }

    private setKeyCooldown(key: string): void {
        this.keyCooldowns[key] = Date.now() + this.COOLDOWN_MS;
    }

    /**
     * 底層 API 調用邏輯 (Unified Fetch)
     */
    private async baseRequest(model: string, apiKey: string, body: any): Promise<any> {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
        
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });

            if (response.status === 404) {
                this.modelAvailability[model] = false; // 標記模型無效
                throw { status: 404, message: `Model ${model} not found` };
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw { 
                    status: response.status, 
                    message: errorData.error?.message || `API Error ${response.status}` 
                };
            }

            return await response.json();
        } catch (e: any) {
            if (!e.status) throw { status: 500, message: e.message || "Network Error" };
            throw e;
        }
    }

    /**
     * 核心排程器：處理轉向、冷卻與失敗重試
     */
    private async scheduledRequest(payloadBuilder: (model: string) => any): Promise<any> {
        const apiKeys = this.getApiKeys();
        if (apiKeys.length === 0) throw { status: 401, message: "No API Keys configured" };

        const testModels = this.preferredModel ? [this.preferredModel, ...GEMINI_MODELS.filter(m => m !== this.preferredModel)] : GEMINI_MODELS;

        for (const model of testModels) {
            if (this.modelAvailability[model] === false) continue; // 跳過已知無效模型

            for (let k = 0; k < apiKeys.length; k++) {
                const keyIdx = (this.apiKeyIndex + k) % apiKeys.length;
                const key = apiKeys[keyIdx];

                if (this.isKeyCoolingDown(key)) continue;

                for (let attempt = 0; attempt < this.MAX_RETRIES; attempt++) {
                    try {
                        console.log(`[LLM] 嘗試 ${model} | Key #${keyIdx + 1}`);
                        const result = await this.baseRequest(model, key, payloadBuilder(model));
                        
                        // 成功，鎖定當前索引以優化下次速度
                        this.apiKeyIndex = keyIdx;
                        return result;
                    } catch (error: any) {
                        console.warn(`[LLM] ${model} 失敗: ${error.message}`);
                        
                        if (error.status === 429) {
                            this.setKeyCooldown(key);
                            break; // 切換 Key
                        }
                        if (error.status === 404) break; // 切換模型
                        if (error.status === 400) break; // 參數錯誤，切換
                        
                        // 其他錯誤（500/網路）重試一次
                        if (attempt === this.MAX_RETRIES - 1) break;
                        await new Promise(r => setTimeout(r, 1000));
                    }
                }
            }
        }
        throw new Error("All nodes and models exhausted");
    }

    /**
     * 計算食材雜湊值作為快取 Key
     */
    private getCacheKey(ingredients: string[], preferences?: string): string {
        const sorted = [...ingredients].sort().join(",");
        return `recipe-cache-${sorted}-${preferences || ""}`;
    }

    async generateRecipes(request: LLMRecipeRequest): Promise<LLMRecipe[]> {
        const cacheKey = this.getCacheKey(request.ingredients, request.preferences);
        
        // 1. 嘗試從快取讀取
        try {
            const cached = localStorage.getItem(cacheKey);
            if (cached) {
                console.log("[LLM] 命中快取，正在載入預存食譜...");
                return JSON.parse(cached);
            }
        } catch (e) {
            console.warn("[LLM] 快取讀取失敗:", e);
        }

        try {
            const data = await this.scheduledRequest((model) => ({
                contents: [{ parts: [{ text: this.getRecipePrompt(request) }] }],
                generationConfig: {
                    temperature: 0.3,
                    responseMimeType: "application/json"
                }
            }));

            const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
            if (!text) throw new Error("Empty AI response");

            const recipes = JSON.parse(text).map((r: any) => ({
                id: `ai-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
                name: r.name || "AI 家常料理",
                image: this.getPlaceholderImage(r.category),
                time: r.time || "20 min",
                difficulty: r.difficulty || "easy",
                category: r.category || "mixed",
                requiredIngredients: this.normalizeArray(r.requiredIngredients || request.ingredients),
                description: r.description || "符合現有食材的實用食譜建議。",
                matchScore: r.matchScore || 90,
                steps: this.normalizeSteps(r.steps),
                sustainabilityTip: r.sustainabilityTip || "",
                substitutionTip: r.substitutionTip || ""
            }));

            // 2. 寫入快取
            try {
                localStorage.setItem(cacheKey, JSON.stringify(recipes));
            } catch (e) {
                console.warn("[LLM] 快取寫入失敗:", e);
            }

            return recipes;
        } catch (e) {
            console.error("[LLM] 生成食譜最終失敗:", e);
            return this.getLocalFallback(request.ingredients);
        }
    }

    private getRecipePrompt(request: LLMRecipeRequest): string {
        let dietaryInfo = "";
        if (request.preferences) {
            try {
                const prefs = JSON.parse(request.preferences);
                const dietaryParts = [];
                if (prefs.vegetarian) dietaryParts.push("素食 (Vegetarian)");
                if (prefs.lowCalorie) dietaryParts.push("減脂/低卡 (Low Calorie / Diet)");
                if (prefs.allergies) dietaryParts.push(`過敏原禁忌：${prefs.allergies}`);
                
                if (dietaryParts.length > 0) {
                    dietaryInfo = `\n**用戶個人偏好設定**：\n${dietaryParts.map(p => `- ${p}`).join("\n")}\n請務必嚴格遵守上述偏好與過敏限制。`;
                }
            } catch (e) {
                console.warn("Failed to parse dietary preferences", e);
            }
        }

        return `你是一位擅長傳統家常菜的資深主廚。
請根據用戶擁有的食材：${request.ingredients.join(", ")}，生成 3 個 JSON 格式的食譜建議。${dietaryInfo}

規則：
1. **風格**：必須是「家常菜」(Home-style)，避免法式精緻料理或過於複雜的創意料理。
2. **食材限制**：盡可能只使用提供的食材。除了常見的基礎調味料（油、鹽、醬油、胡椒、糖、蒜、蔥）外，不可隨意添加其他需要額外購買的主菜或配料。
3. **續航與替代**：如果必須使用到某些常用但用戶沒提到的食材（例如雞蛋、洋蔥），請在 \`substitutionTip\` 中說明替代方案。
4. **零浪費智慧 (Zero-Waste)**：你必須針對這道食譜提供一個 \`sustainabilityTip\`，告訴用戶如何減少食材浪費（例如：剩餘菜梗可以熬湯、果皮可以除臭、如何延長食材保存等）。
5. **輸出格式**：必須維持 JSON Array 格式。
格式：[{name, time, difficulty, category, requiredIngredients, description, sustainabilityTip, substitutionTip, steps:[{title, description}], matchScore}]`;
    }

    private cleanJson(text: string): string {
        // 移除 Markdown 代碼塊 (如 ```json ... ```)
        return text.replace(/```json/g, "").replace(/```/g, "").trim();
    }

    async detectIngredientsFromImage(base64Image: string): Promise<any[]> {
        const data = await this.scheduledRequest((model) => ({
            contents: [{
                parts: [
                    { text: "分析這張照片，辨識所有食材。請僅回傳 JSON Array 格式，包含: [{name, category, isSpoiled, confidence}]。不要有任何解釋文字。" },
                    { inline_data: { mime_type: "image/jpeg", data: base64Image } }
                ]
            }],
            generationConfig: { 
                temperature: 0.2, 
                responseMimeType: "application/json" 
            }
        }));

        const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) throw new Error("Vision response empty (視覺辨識反應為空)");

        try {
            const cleaned = this.cleanJson(text);
            const results = JSON.parse(cleaned);
            return results.map((r: any) => ({
                id: `ai-vision-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
                name: r.name,
                quantity: 1,
                category: r.category || "其他",
                timestamp: Date.now(),
                confidence: r.confidence || 0.9,
                isSpoiled: r.isSpoiled || false,
                storageType: "fridge"
            }));
        } catch (e) {
            console.error("[LLM] Vision JSON Parse Error:", e, "Raw text:", text);
            throw new Error("無法解析識別結果，AI 反應格式不正確。");
        }
    }

    private getPlaceholderImage(category: string): string {
        const images: Record<string, string> = {
            vegetable: "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=800",
            meat: "https://images.unsplash.com/photo-1544025162-d76694265947?q=80&w=800",
            mixed: "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800",
            fruit: "https://images.unsplash.com/photo-1519996529931-28324d5a630e?q=80&w=800"
        };
        return images[category] || images.mixed;
    }

    private getLocalFallback(ingredients: string[]): LLMRecipe[] {
        const templates = [
            { name: `${ingredients[0] || "綜合"}風味炒飯`, category: "mixed" as const },
            { name: `家常 ${ingredients[0] || "精選"} 濃湯`, category: "vegetable" as const },
            { name: `活力 ${ingredients[0] || "低碳"} 沙拉`, category: "mixed" as const }
        ];

        return templates.map((t, idx) => ({
            id: `local-${idx}-${Date.now()}`,
            name: t.name,
            image: this.getPlaceholderImage(t.category),
            time: "15 min",
            difficulty: "easy",
            category: t.category,
            requiredIngredients: ingredients,
            description: "本地端智慧匹配生成的食譜方案。",
            matchScore: 90,
            steps: [{ title: "準備", description: "清洗並準備所有食材。" }, { title: "烹飪", description: "依照個人喜好進行烹調。" }]
        }));
    }

    getMaskedApiKeys(): string[] {
        return this.getApiKeys().map(key => {
            if (key.length <= 8) return "****";
            return `${key.slice(0, 4)}...${key.slice(-4)}`;
        });
    }

    getKeyStatusList(): { masked: string; coolingDown: boolean; cooldownRemaining: number; isCustom: boolean }[] {
        const customKeys = this.customApiKeys.map(k => k.trim());
        return this.getApiKeys().map(key => {
            const coolingDown = this.isKeyCoolingDown(key);
            const cooldownRemaining = coolingDown
                ? Math.ceil((this.keyCooldowns[key] - Date.now()) / 1000)
                : 0;
            const masked = key.length <= 8 ? "****" : `${key.slice(0, 4)}...${key.slice(-4)}`;
            const isCustom = customKeys.includes(key);
            return { masked, coolingDown, cooldownRemaining, isCustom };
        });
    }

    getActiveKeyInfo(): string | null {
        const apiKeys = this.getApiKeys();
        if (apiKeys.length === 0) return null;
        const key = apiKeys[this.apiKeyIndex];
        return key.length <= 8 ? "****" : `${key.slice(0, 4)}...${key.slice(-4)}`;
    }

    async testConnection(): Promise<{ status: 'online' | 'offline' | 'no_key', model: string, keyCount: number, activeKey: string | null }> {
        const keys = this.getApiKeys();
        if (keys.length === 0) return { status: 'no_key', model: GEMINI_MODELS[0], keyCount: 0, activeKey: null };

        try {
            // Test with a very small request
            await this.baseRequest(GEMINI_MODELS[0], keys[0], {
                contents: [{ parts: [{ text: "ping" }] }],
                generationConfig: { maxOutputTokens: 1 }
            });
            return { status: 'online', model: GEMINI_MODELS[0], keyCount: keys.length, activeKey: this.getActiveKeyInfo() };
        } catch (e) {
            console.warn("[LLM] Connection test failed:", e);
            return { status: 'offline', model: GEMINI_MODELS[0], keyCount: keys.length, activeKey: null };
        }
    }

    getAvailableModels() {
        return GEMINI_MODELS;
    }

    /**
     * 輔助方法：確保資料為陣列格式
     */
    private normalizeArray(data: any): string[] {
        if (Array.isArray(data)) return data.map(String);
        if (typeof data === 'string') return data.split(/[,，\s]+/).filter(Boolean);
        return [];
    }

    private normalizeSteps(steps: any): { title: string; description: string }[] {
        if (Array.isArray(steps)) {
            return steps.map(s => ({
                title: String(s.title || "步驟"),
                description: String(s.description || "")
            }));
        }
        if (typeof steps === 'string') {
            return steps.split('\n').filter(Boolean).map(line => ({
                title: "烹飪步驟",
                description: line.trim()
            }));
        }
        return [{ title: "準備", description: "依照直覺進行烹飪。" }];
    }
}

export const llmService = new LLMService();

