import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { llmService } from "./llmService";
import { yoloService } from "./yoloService";
import { notificationService } from "./notificationService";

/**
 * 介面 (Interface): 單件掃描食材 (ScannedItem)
 * 用於規範每一個被系統記錄或 YOLO 辨識出來的食材結構。
 */
export interface ScannedItem {
    id: string;
    name: string;
    quantity: number;
    timestamp: number;
    category?: string;
    freshness?: number; // 0-10
    expiryDays?: number; // Days until expiry
    confidence?: number;
    isSpoiled?: boolean;
    box?: number[]; // [x1, y1, x2, y2]
    storageType?: "fridge" | "freezer";
}

export interface WasteRecord {
    date: string;
    amount: number; // in grams or items
    items?: string[]; // 具體浪費的食材名稱清單
}

interface IngredientContextType {
    scannedItems: ScannedItem[];
    recommendedRecipes: any[];
    tempDetections: ScannedItem[];
    selectedIds: string[];
    settings: { 
        notifications: boolean; 
        neuralOptimized: boolean; 
        confidenceThreshold: number; 
        darkMode: boolean;
        dietary: {
            vegetarian: boolean;
            lowCalorie: boolean;
            allergies: string;
        };
        uiScale: number;
        autoScale: boolean;
        customApiKeys: string;
        themeColor: string;
        isModalOpen: boolean;
    };
    wasteHistory: WasteRecord[];
    savedRecipes: any[];
    addItem: (item: Partial<ScannedItem>, source?: "ai" | "manual") => void;
    updateQuantity: (id: string, delta: number) => void;
    updateItem: (id: string, updates: Partial<ScannedItem>) => void;
    removeItem: (id: string) => void;
    removeIngredient: (id: string) => void;
    toggleSelection: (id: string) => void;
    generateRecipe: () => Promise<void>;
    saveRecipe: (recipe: any) => void;
    unsaveRecipe: (recipeId: string) => void;
    clearAll: () => void;
    clearInventory: () => void;
    clearWasteHistory: () => void;
    resetSettings: () => void;
    setRecipes: (recipes: any[]) => void;
    clearTempDetections: () => void;
    updateSettings: (settings: Partial<IngredientContextType['settings']>) => void;
}

/**
 * 上下文宣告 (Context Context): IngredientContext
 * 用來實現跨元件共享狀態 (Global State)，避免 Props Drilling。
 */
const IngredientContext = createContext<IngredientContextType | undefined>(undefined);

/**
 * 狀態池提供者 (Provider): IngredientProvider
 * 必須包在應用程式的最外層 (如 App.tsx)，它持有並管理所有核心業務資料：
 * 1. 冰箱內所有庫存 (scannedItems)
 * 2. 暫存辨識清單 (tempDetections)
 * 3. 系統全域設定 (settings)
 * 4. 食譜庫 (recommendedRecipes)
 */
export function IngredientProvider({ children }: { children: ReactNode }) {
    const [scannedItems, setScannedItems] = useState<ScannedItem[]>([]);
    const [recommendedRecipes, setRecommendedRecipes] = useState<any[]>([]);
    const [tempDetections, setTempDetections] = useState<ScannedItem[]>([]);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [settings, setSettings] = useState({ 
        notifications: true, 
        neuralOptimized: true, 
        confidenceThreshold: 0.25, 
        darkMode: true,
        dietary: {
            vegetarian: false,
            lowCalorie: false,
            allergies: ""
        },
        uiScale: 1.0,
        autoScale: true,
        customApiKeys: "",
        themeColor: "#00ff88",
        isModalOpen: false
    });
    const [savedRecipes, setSavedRecipes] = useState<any[]>([]);

    const [wasteHistory, setWasteHistory] = useState<WasteRecord[]>([]);

    // Load from localStorage on mount & Pre-warm YOLO
    useEffect(() => {
        yoloService.prewarm(); // [Optimized] 全域自動預熱，確保進入掃描頁時已就緒
        
        const saved = localStorage.getItem("scannedIngredients");
        const savedRecs = localStorage.getItem("recommendedRecipes");
        const savedSettings = localStorage.getItem("appSettings");
        const savedBookmarked = localStorage.getItem("savedRecipes");
        const savedWaste = localStorage.getItem("wasteHistory");
        
        if (saved) try { setScannedItems(JSON.parse(saved)); } catch (e) { setScannedItems([]); }
        if (savedRecs) try { setRecommendedRecipes(JSON.parse(savedRecs)); } catch (e) { setRecommendedRecipes([]); }
        if (savedSettings) {
            try { 
                const parsed = JSON.parse(savedSettings);
                // Migration: Ensure dietary settings exist
                if (!parsed.dietary) {
                    parsed.dietary = { vegetarian: false, lowCalorie: false, allergies: "" };
                }
                if (parsed.uiScale === undefined) parsed.uiScale = 1.0;
                if (parsed.autoScale === undefined) parsed.autoScale = true;
                if (parsed.customApiKeys === undefined) parsed.customApiKeys = "";
                if (parsed.themeColor === undefined || parsed.themeColor === "var(--primary-default)") parsed.themeColor = "#00ff88";
                setSettings(parsed); 
            } catch (e) { }
        }
        if (savedBookmarked) try { setSavedRecipes(JSON.parse(savedBookmarked)); } catch (e) { setSavedRecipes([]); }
        if (savedWaste) {
            try { 
                const parsed = JSON.parse(savedWaste);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    setWasteHistory(parsed); 
                } else {
                    setWasteHistory([]);
                }
            } catch (e) { setWasteHistory([]); }
        } else {
            setWasteHistory([]);
        }
    }, []);

    // Notification check effect
    useEffect(() => {
        if (settings.notifications && scannedItems.length > 0) {
            // 延遲一點執行，避免剛載入時太突兀
            const timer = setTimeout(() => {
                notificationService.checkAndNotify(scannedItems, settings);
            }, 3000);
            return () => clearTimeout(timer);
        }
    }, [scannedItems, settings.notifications]);

    // Sync custom API keys to LLM service
    useEffect(() => {
        if (settings.customApiKeys !== undefined) {
            llmService.setCustomApiKeys(settings.customApiKeys);
        }
    }, [settings.customApiKeys]);

    /** 同步狀態至本地端 localStorage，實現離線存取與關閉保留 (Data Persistence) */
    useEffect(() => {
        localStorage.setItem("scannedIngredients", JSON.stringify(scannedItems));
    }, [scannedItems]);

    useEffect(() => {
        localStorage.setItem("recommendedRecipes", JSON.stringify(recommendedRecipes));
    }, [recommendedRecipes]);

    useEffect(() => {
        localStorage.setItem("appSettings", JSON.stringify(settings));
    }, [settings]);

    useEffect(() => {
        localStorage.setItem("savedRecipes", JSON.stringify(savedRecipes));
    }, [savedRecipes]);

    useEffect(() => {
        localStorage.setItem("wasteHistory", JSON.stringify(wasteHistory));
    }, [wasteHistory]);

    /**
     * 新增單一食材進入庫存
     * 實施「智慧批次合併邏輯」：
     * 1. 只有當食材名稱、儲存位置相同，且登錄時間相差在 1 小時內時，才會進行合併。
     * 2. 超過 1 小時的新增將視為「新採購批次」，獨立顯示以確保食品安全。
     */
    const addItem = (item: Partial<ScannedItem>, source: "ai" | "manual" = "ai") => {
        const now = Date.now();
        const BATCH_THRESHOLD = 3600000; // 1 小時 (毫秒)
        const uniqueId = item.id || `${now}-${Math.random().toString(36).substr(2, 9)}`;
        
        const newItem: ScannedItem = {
            id: uniqueId,
            name: item.name || "未知食材",
            quantity: item.quantity || 1,
            timestamp: item.timestamp || now,
            category: item.category || "其他",
            storageType: item.storageType || "fridge",
            expiryDays: item.expiryDays !== undefined ? item.expiryDays : 7,
            ...item
        };

        setScannedItems(prev => {
            // 尋找符合合併條件的既存項目：名稱相同、品質相同、位置相同、且時間在一小時內
            const existing = prev.find(i => 
                i.name.toLowerCase() === newItem.name.toLowerCase() && 
                i.isSpoiled === newItem.isSpoiled && 
                i.storageType === newItem.storageType &&
                Math.abs(now - (i.timestamp || now)) < BATCH_THRESHOLD
            );

            if (existing) {
                return prev.map(i =>
                    i.id === existing.id
                        ? { ...i, quantity: i.quantity + newItem.quantity, timestamp: now } // 更新數量並刷新時間（同批次）
                        : i
                );
            }
            return [...prev, newItem]; // 不同批次，獨立新增
        });

        // 同步更新暫存辨識清單 (僅限 AI 模式)
        if (source === "ai") {
            setTempDetections(prev => {
                const existing = prev.find(i => 
                    i.name.toLowerCase() === newItem.name.toLowerCase() && 
                    i.isSpoiled === newItem.isSpoiled && 
                    i.storageType === newItem.storageType &&
                    Math.abs(now - (i.timestamp || now)) < BATCH_THRESHOLD
                );
                
                if (existing) {
                    return prev.map(i =>
                        i.id === existing.id
                            ? { ...i, quantity: i.quantity + newItem.quantity, timestamp: now }
                            : i
                    );
                }
                return [...prev, newItem];
            });
        }

        // Auto-select ONLY IF NOT SPOILED
        if (!newItem.isSpoiled) {
            setSelectedIds(prev => {
                const existingItem = scannedItems.find(i => i.name.toLowerCase() === newItem.name.toLowerCase() && i.isSpoiled === newItem.isSpoiled && i.storageType === newItem.storageType);
                const targetId = existingItem ? existingItem.id : uniqueId;
                return prev.includes(targetId) ? prev : [...prev, targetId];
            });
        }
    };

    const clearTempDetections = () => setTempDetections([]);

    const updateQuantity = (id: string, delta: number) => {
        const updater = (prev: ScannedItem[]) =>
            prev.map(item =>
                item.id === id ? { ...item, quantity: Math.max(0, item.quantity + delta) } : item
            ).filter(item => item.quantity > 0);

        setScannedItems(updater);
        setTempDetections(updater);
    };

    const updateItem = (id: string, updates: Partial<ScannedItem>) => {
        setScannedItems(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
        setTempDetections(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
    };

    const removeItem = (id: string) => {
        if (!id) return;
        
        const itemToRemove = scannedItems.find(i => i.id === id);
        
        if (itemToRemove) {
            const now = Date.now();
            const daysPassed = Math.floor((now - (itemToRemove.timestamp || now)) / (1000 * 60 * 60 * 24));
            const expiryDays = itemToRemove.expiryDays !== undefined ? itemToRemove.expiryDays : 7;
            const daysLeft = expiryDays - daysPassed;
            
            // 判定條件：已標記損壞、天數到期、或手動設為 0
            const isWaste = itemToRemove.isSpoiled || daysLeft <= 0 || expiryDays <= 0;
            
            if (isWaste) {
                // 發送即時通知，告知數據已錄入
                notificationService.send("📊 數據統計更新", `已將 "${itemToRemove.name}" 計入今日浪費數據`);

                setWasteHistory(prev => {
                    const today = new Date();
                    // 格式：YYYY-MM-DD (精確到本地日期)
                    const dateStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
                    
                    const newHistory = [...prev];
                    const existingIdx = newHistory.findIndex(h => h.date === dateStr);
                    
                    if (existingIdx !== -1) {
                        const existing = newHistory[existingIdx];
                        newHistory[existingIdx] = {
                            ...existing,
                            amount: (Number(existing.amount) || 0) + 1,
                            items: Array.from(new Set([...(existing.items || []), itemToRemove.name]))
                        };
                    } else {
                        newHistory.push({
                            date: dateStr,
                            amount: 1,
                            items: [itemToRemove.name]
                        });
                        newHistory.sort((a, b) => a.date.localeCompare(b.date));
                    }
                    return newHistory;
                });
            }
        }

        setScannedItems(prev => prev.filter(item => item.id !== id));
        setTempDetections(prev => prev.filter(item => item.id !== id));
        setSelectedIds(prev => prev.filter(selectedId => selectedId !== id));
    };

    const removeIngredient = removeItem;

    const generateRecipe = async () => {
        const selectedIngredients = scannedItems
            .filter(item => selectedIds.includes(item.id) && !item.isSpoiled)
            .map(item => item.name);

        if (selectedIngredients.length === 0) {
            throw new Error("請選擇有效的食材進行合成（損壞食材將自動排除）");
        }

        const recipes = await llmService.generateRecipes({ 
            ingredients: selectedIngredients,
            preferences: JSON.stringify(settings.dietary) 
        });
        setRecommendedRecipes(recipes);
    };

    const toggleSelection = (id: string) => {
        const item = scannedItems.find(i => i.id === id);
        if (item?.isSpoiled) return; // Cannot select spoiled items
        setSelectedIds(prev =>
            prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
        );
    };

    const clearAll = () => {
        localStorage.clear();
        setScannedItems([]);
        setRecommendedRecipes([]);
        setTempDetections([]);
        setSavedRecipes([]);
        setWasteHistory([]);
        setSelectedIds([]);
        window.location.reload();
    };

    const clearInventory = () => {
        localStorage.removeItem("scannedIngredients");
        localStorage.removeItem("tempDetections");
        setScannedItems([]);
        setTempDetections([]);
        notificationService.send("🧹 系統清理", "已清空所有庫存資料");
    };

    const clearWasteHistory = () => {
        localStorage.removeItem("wasteHistory");
        setWasteHistory([]);
        notificationService.send("🧹 系統清理", "已清空所有浪費統計數據");
    };

    const resetSettings = () => {
        const defaults = { 
            notifications: true, 
            neuralOptimized: true, 
            confidenceThreshold: 0.25, 
            darkMode: true,
            dietary: { vegetarian: false, lowCalorie: false, allergies: "" },
            uiScale: 1.0,
            autoScale: true,
            customApiKeys: settings.customApiKeys, // 保留當前金鑰，避免用戶重複輸入
            themeColor: "#00ff88",
            isModalOpen: false
        };
        setSettings(defaults);
        localStorage.setItem("appSettings", JSON.stringify(defaults));
        notificationService.send("⚙️ 系統還原", "已將設定還原為初始狀態 (金鑰已保留)");
    };

    const updateSettings = (newSettings: Partial<IngredientContextType['settings']>) => {
        setSettings(prev => ({ ...prev, ...newSettings }));
    };

    const saveRecipe = (recipe: any) => {
        setSavedRecipes(prev => {
            if (prev.find(r => r.id === recipe.id)) return prev;
            return [...prev, recipe];
        });
    };

    const unsaveRecipe = (recipeId: string) => {
        setSavedRecipes(prev => prev.filter(r => r.id !== recipeId));
    };

    return (
        <IngredientContext.Provider value={{
            scannedItems,
            recommendedRecipes,
            tempDetections,
            selectedIds,
            settings,
            wasteHistory,
            addItem,
            updateQuantity,
            updateItem,
            removeItem,
            removeIngredient,
            toggleSelection,
            generateRecipe,
            saveRecipe,
            unsaveRecipe,
            savedRecipes,
            clearAll,
            clearInventory,
            clearWasteHistory,
            resetSettings,
            setRecipes: setRecommendedRecipes,
            clearTempDetections,
            updateSettings
        }}>
            {children}
        </IngredientContext.Provider>
    );
}

export function useIngredients() {
    const context = useContext(IngredientContext);
    if (context === undefined) {
        throw new Error("useIngredients must be used within an IngredientProvider");
    }
    return context;
}
