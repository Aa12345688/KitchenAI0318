import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Settings, Sparkles, ChefHat, Palette, Bell, Moon, AlertTriangle, Loader2, Package, RefreshCw, LogOut, ChevronRight } from "lucide-react";
import { llmService } from "../../services/llmService";
import { notificationService } from "../../services/notificationService";

interface SettingsModalProps {
    type: string;
    onClose: () => void;
    settings: any;
    updateSettings: (s: any) => void;
    apiStatus: any;
    onClearInventory?: () => void;
    onClearWaste?: () => void;
    onResetSettings?: () => void;
    onClearAll?: () => void;
}

export function SettingsModal({ 
    type, onClose, settings, updateSettings, apiStatus,
    onClearInventory, onClearWaste, onResetSettings, onClearAll 
}: SettingsModalProps) {
    const [testRunning, setTestRunning] = useState(false);
    const [keyInput, setKeyInput] = useState(settings?.customApiKeys || "");

    const titles: any = {
        api: "神經節點配置",
        dietary: "飲食偏好",
        theme: "外觀風格",
        display: "介面縮放",
        system: "通知與設定",
        data: "數據與隱私管理"
    };

    const handleKeySave = () => {
        updateSettings({ customApiKeys: keyInput });
        notificationService.send("金鑰已更新", "系統核心已重新掛載");
    };

    const handleTest = async () => {
        setTestRunning(true);
        const res = await llmService.testConnection();
        setTestRunning(false);
        if (res.status === 'online') {
            notificationService.send("連線成功", `模型: ${res.model}`);
        } else {
            notificationService.send("連線失敗", "請檢查網路與 API 金鑰");
        }
    };

    return (
        <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }} 
            onPointerDown={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            className="fixed inset-0 z-[200] bg-background/95 backdrop-blur-3xl flex items-center justify-center p-4"
        >
            <motion.div 
                initial={{ scale: 0.95, y: 20 }} 
                animate={{ scale: 1, y: 0 }} 
                onPointerDown={(e) => e.stopPropagation()}
                className="bg-[#1a4d3d] w-[92vw] max-w-[420px] rounded-[2rem] border border-white/10 shadow-2xl relative overflow-hidden flex flex-col max-h-[92vh]"
            >
                <div className="absolute top-0 right-0 p-4 z-20">
                    <button onClick={onClose} className="w-[2.2rem] h-[2.2rem] rounded-full bg-white/5 flex items-center justify-center text-gray-500 hover:text-white transition-all"><X style={{ width: '1.2rem', height: '1.2rem' }} /></button>
                </div>

                <div className="px-[1.5rem] pt-[2rem] pb-[1rem] shrink-0 border-b border-white/5 mb-4">
                    <div className="flex items-center gap-4">
                        <div className="w-[3rem] h-[3rem] bg-primary/10 rounded-2xl flex items-center justify-center shadow-2xl border border-primary/20 shrink-0">
                            <Settings style={{ width: '1.5rem', height: '1.5rem' }} className="text-primary" />
                        </div>
                        <div className="min-w-0">
                            <h3 className="text-[1.2rem] font-black text-white uppercase tracking-tighter truncate">{titles[type]}</h3>
                            <p className="text-[0.45rem] font-bold text-primary/60 uppercase tracking-widest mt-0.5 opacity-80">Settings & Configuration</p>
                        </div>
                    </div>
                </div>

                <div className="px-[1.5rem] pb-[1rem] overflow-y-auto w-full no-scrollbar flex-1 space-y-6">
                    {type === 'api' && (
                        <div className="space-y-6">
                            <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="text-[10px] font-black text-white uppercase tracking-widest">核心節點狀態</div>
                                    <div className={`px-2 py-1 rounded-full text-[8px] font-black uppercase ${apiStatus?.status === 'online' ? 'bg-primary/20 text-primary' : 'bg-red-500/20 text-red-500'}`}>
                                        {apiStatus?.status === 'online' ? 'Online' : 'Offline'}
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    {[{label: '預設模型', val: apiStatus?.model || 'None'}, {label: '連線狀態', val: apiStatus?.status === 'online' ? '穩定' : '離線'}].map((item, i) => (
                                        <div key={i} className="bg-black/20 p-3 rounded-xl">
                                            <div className="text-[7px] font-bold text-gray-500 uppercase mb-1">{item.label}</div>
                                            <div className="text-[9px] font-black text-white truncate">{item.val}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <div className="flex justify-between items-end mb-2">
                                    <label className="text-[9px] font-black text-gray-500 uppercase tracking-widest block">Google Gemini API KEY</label>
                                    <div className="text-[8px] font-bold text-primary/60 bg-primary/10 px-2 py-0.5 rounded-full border border-primary/20">
                                        已偵測到 {llmService.getMaskedApiKeys().length} 組金鑰
                                    </div>
                                </div>
                                <div className="relative">
                                    <input 
                                        type="password" 
                                        value={keyInput}
                                        onChange={(e) => setKeyInput(e.target.value)}
                                        placeholder="KEY1, KEY2, KEY3..."
                                        className="w-full bg-black/40 border border-white/10 rounded-2xl px-5 py-4 text-white font-bold text-sm focus:border-primary outline-none transition-all placeholder:text-gray-700" 
                                    />
                                    {keyInput && <button onClick={handleKeySave} className="absolute right-3 top-1/2 -translate-y-1/2 bg-primary text-background text-[8px] font-black px-3 py-2 rounded-xl uppercase tracking-widest active:scale-95 transition-all">更新</button>}
                                </div>
                                <p className="text-[8px] font-bold text-gray-500 uppercase mt-2 px-1">💡 提示：輸入多組金鑰（以逗號隔開）可提升辨識配額並避免 429 錯誤。</p>
                            </div>

                            <button
                                onClick={handleTest}
                                disabled={testRunning}
                                className="w-full h-14 bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center gap-3 hover:bg-white/10 transition-all font-black text-xs uppercase tracking-widest text-white disabled:opacity-50"
                            >
                                {testRunning ? <Loader2 className="animate-spin" size={18} /> : <Sparkles className="text-primary" size={18} />}
                                {testRunning ? "正在驗證節點連線..." : "測試 API 通訊能力"}
                            </button>
                        </div>
                    )}

                    {type === 'dietary' && (
                        <div className="space-y-6">
                            {['偏好', '過敏'].map((title, i) => (
                                <div key={i}>
                                    <label className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-3 block">{title}</label>
                                    <div className="flex flex-wrap gap-2">
                                        {(i === 0 ? ['素食', '低卡'] : ['海鮮', '堅果', '乳製品']).map(tag => {
                                            const isActive = i === 0 
                                                ? (tag === '素食' ? settings?.dietary?.vegetarian : settings?.dietary?.lowCalorie)
                                                : settings?.dietary?.allergies?.includes(tag);
                                            
                                            return (
                                                <button 
                                                    key={tag} 
                                                    onClick={() => {
                                                        if (i === 0) {
                                                            const key = tag === '素食' ? 'vegetarian' : 'lowCalorie';
                                                            updateSettings({ dietary: { ...settings.dietary, [key]: !settings.dietary[key] } });
                                                        } else {
                                                            let allergies = settings.dietary.allergies || "";
                                                            if (allergies.includes(tag)) {
                                                                allergies = allergies.split(', ').filter((t: string) => t !== tag).join(', ');
                                                            } else {
                                                                allergies = allergies ? `${allergies}, ${tag}` : tag;
                                                            }
                                                            updateSettings({ dietary: { ...settings.dietary, allergies } });
                                                        }
                                                    }}
                                                    className={`px-4 py-2 rounded-xl text-[10px] font-black uppercase transition-all border ${isActive ? 'bg-primary border-primary text-background' : 'bg-white/5 border-white/10 text-gray-400'}`}
                                                >
                                                    {tag}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {type === 'theme' && (
                        <div className="space-y-6">
                            <div className="grid grid-cols-3 gap-3">
                                {['#00ff88', '#60a5fa', '#f87171', '#fbbf24', '#c084fc', '#ffffff'].map(c => (
                                    <button 
                                        key={c} 
                                        onClick={() => updateSettings({ themeColor: c })}
                                        className="group relative aspect-square rounded-2xl border border-white/10 transition-all flex items-center justify-center overflow-hidden active:scale-90 shadow-xl"
                                        style={{ backgroundColor: `${c}10` }}
                                    >
                                        <div className="w-6 h-6 rounded-full border-2 border-white/20 relative z-10 shadow-lg transition-transform group-hover:scale-110" style={{ backgroundColor: c }} />
                                        {settings?.themeColor === c && <div className="absolute inset-0 bg-white/10 animate-pulse" />}
                                    </button>
                                ))}
                            </div>
                            <div className="bg-black/40 rounded-3xl p-6 border border-white/10 flex items-center justify-between">
                                <div>
                                    <div className="text-xs font-black text-white uppercase tracking-widest">深色模式動態切換</div>
                                    <div className="text-[8px] font-bold text-gray-500 uppercase mt-1">Smart Atmosphere Sync</div>
                                </div>
                                <button onClick={() => updateSettings({ darkMode: !settings.darkMode })} className={`w-14 h-8 rounded-full relative p-1 transition-all ${settings.darkMode ? 'bg-primary' : 'bg-gray-700'}`}>
                                    <div className={`w-6 h-6 bg-background rounded-full flex items-center justify-center shadow-lg transition-all ${settings.darkMode ? 'translate-x-6' : 'translate-x-0'}`}><Moon size={12} className={settings.darkMode ? "text-primary" : "text-gray-400"} /></div>
                                </button>
                            </div>
                        </div>
                    )}

                    {type === 'display' && (
                        <div className="space-y-8">
                            <div className="bg-black/40 rounded-3xl p-6 border border-white/10 flex items-center justify-between">
                                <div>
                                    <div className="text-xs font-black text-white uppercase tracking-widest">自動螢幕適應</div>
                                    <div className="text-[8px] font-bold text-gray-500 uppercase mt-1">Auto-detect best zoom</div>
                                </div>
                                <button onClick={() => updateSettings({ autoScale: !settings?.autoScale })} className={`w-14 h-8 rounded-full relative p-1 transition-all ${settings?.autoScale ? 'bg-primary' : 'bg-gray-700'}`}>
                                    <div className={`w-6 h-6 bg-background rounded-full flex items-center justify-center shadow-lg transition-all ${settings?.autoScale ? 'translate-x-6' : 'translate-x-0'}`}><Sparkles size={12} className={settings?.autoScale ? "text-primary" : "text-gray-400"} /></div>
                                </button>
                            </div>

                             <div 
                                className={settings?.autoScale ? "opacity-40 pointer-events-none" : "relative z-[300] touch-none py-2"}
                                onPointerDown={(e) => {
                                    e.stopPropagation();
                                }}
                             >
                                <div className="flex justify-between items-center mb-4">
                                    <label className="text-[9px] font-black text-gray-500 uppercase tracking-widest">手動介面縮放 (Manual Scaling)</label>
                                    <span className="text-xs font-black text-primary">{Math.round((settings?.uiScale || 1.0) * 100)}%</span>
                                </div>
                                <div className="px-1 relative z-[300]">
                                    <input 
                                        type="range" 
                                        min="1.0" 
                                        max="2.0" 
                                        step="0.1" 
                                        value={settings?.uiScale || 1.0}
                                        onPointerDown={(e) => e.stopPropagation()}
                                        onTouchStart={(e) => e.stopPropagation()}
                                        onChange={(e) => updateSettings({ uiScale: parseFloat(e.target.value), autoScale: false })}
                                        className="w-full h-1.5 bg-black/40 rounded-full appearance-none accent-primary cursor-pointer border border-white/5" 
                                        style={{ touchAction: 'none' }}
                                    />
                                </div>
                                <div className="flex justify-between mt-3 px-1">
                                    <span className="text-[8px] font-black text-gray-700 uppercase">標準視網膜</span>
                                    <span className="text-[8px] font-black text-gray-700 uppercase">舒適閱讀</span>
                                    <span className="text-[8px] font-black text-gray-700 uppercase">超級放大鏡</span>
                                </div>
                            </div>
                            
                            <div className="bg-primary/5 border border-primary/20 rounded-2xl p-4 flex gap-3">
                                <AlertTriangle size={18} className="text-primary shrink-0" />
                                <p className="text-[10px] text-gray-400 font-bold leading-relaxed uppercase">
                                    提示：開啟「智慧寬度自適應」將由系統根據螢幕解析度自動決定最佳渲染比例。手動調整將覆蓋此設定。
                                </p>
                            </div>
                        </div>
                    )}

                    {type === 'system' && (
                        <div className="space-y-4">
                            {[
                                { title: '食材到期提醒', desc: '食材快過期時通知我', icon: Bell, key: 'notifications' },
                                { title: '清單自動化優化', desc: '自動修正並整理食材資訊', icon: Sparkles, key: 'neuralOptimized' }
                            ].map((sys, i) => (
                                <button 
                                    key={i} 
                                    onClick={() => updateSettings({ [sys.key]: !settings[sys.key] })}
                                    className="w-full bg-black/40 hover:bg-black/60 border border-white/10 rounded-2xl p-5 flex items-center justify-between transition-all group relative overflow-hidden"
                                >
                                    <div className="flex items-center gap-4 relative z-10">
                                        <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center border border-white/10 group-hover:border-primary/50 transition-all">
                                            <sys.icon size={18} className={settings[sys.key] ? "text-primary" : "text-gray-400"} />
                                        </div>
                                        <div className="text-left">
                                            <div className="text-xs font-black text-white uppercase tracking-widest">{sys.title}</div>
                                            <div className="text-[8px] font-bold text-gray-500 uppercase mt-0.5">{sys.desc}</div>
                                        </div>
                                    </div>
                                    <div className={`w-10 h-6 rounded-full relative p-0.5 border transition-all ${settings[sys.key] ? 'bg-primary/20 border-primary/30' : 'bg-gray-800 border-white/5'}`}>
                                        <div className={`w-4.5 h-4.5 bg-primary rounded-full shadow-lg transition-all ${settings[sys.key] ? 'translate-x-4' : 'translate-x-0'}`} />
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}

                    {type === 'data' && (
                        <div className="space-y-4">
                            <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-4 flex gap-3 mb-2">
                                <AlertTriangle size={18} className="text-red-500 shrink-0" />
                                <p className="text-[10px] text-red-500/60 font-bold leading-relaxed uppercase">
                                    注意：以下操作涉及敏感資料。一旦執行，對應的數據將從本地存儲中永久移除。
                                </p>
                            </div>

                            {[
                                { 
                                    title: '清空食材庫存', 
                                    desc: '移除所有冰箱內的食材紀錄', 
                                    icon: Package, 
                                    action: () => { if(window.confirm("確定要清空庫存嗎？")) { onClearInventory?.(); onClose(); } }
                                },
                                { 
                                    title: '清空浪費統計', 
                                    desc: '移除歷史耗損數據與圓餅圖紀錄', 
                                    icon: AlertTriangle, 
                                    action: () => { if(window.confirm("確定要清空浪費統計嗎？")) { onClearWaste?.(); onClose(); } }
                                },
                                { 
                                    title: '還原初始設定', 
                                    desc: '重置 UI 縮放與偏好，並保留庫存與 API 金鑰', 
                                    icon: RefreshCw, 
                                    action: () => { if(window.confirm("確定要重置 UI 設定嗎？(食材庫存與金鑰會保留)")) { onResetSettings?.(); onClose(); } }
                                },
                                { 
                                    title: '全面清除所有數據', 
                                    desc: '徹底清空 App 所有紀錄與 Google API 金鑰', 
                                    variant: 'danger',
                                    icon: LogOut, 
                                    action: () => { if(window.confirm("⚠️ 警告：這將徹底清除所有資料包含 API KEY，需要重新設定 App，確定要執行嗎？")) { onClearAll?.(); } }
                                }
                            ].map((item, i) => (
                                <button 
                                    key={i} 
                                    onClick={item.action}
                                    className={`w-full ${item.variant === 'danger' ? 'bg-red-500/5 border-red-500/20' : 'bg-black/40 border-white/10'} hover:brightness-125 border rounded-2xl p-5 flex items-center justify-between transition-all group`}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className={`w-10 h-10 rounded-xl ${item.variant === 'danger' ? 'bg-red-500/20' : 'bg-white/5'} flex items-center justify-center border border-white/10 group-hover:border-primary/50 transition-all`}>
                                            <item.icon size={18} className={item.variant === 'danger' ? "text-red-500" : "text-gray-400"} />
                                        </div>
                                        <div className="text-left">
                                            <div className={`text-xs font-black uppercase tracking-widest ${item.variant === 'danger' ? 'text-red-500' : 'text-white'}`}>{item.title}</div>
                                            <div className="text-[8px] font-bold text-gray-500 uppercase mt-0.5">{item.desc}</div>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-gray-600 group-hover:text-white" />
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="p-[1.2rem] pt-0 shrink-0">
                    <button 
                        onClick={onClose} 
                        className="w-full bg-primary text-black py-[0.8rem] rounded-[1.2rem] font-black uppercase text-[0.85rem] tracking-tight shadow-[0_10px_30px_rgba(0,255,136,0.2)] hover:scale-[1.02] active:scale-[0.98] transition-all"
                    >
                        保存並關閉
                    </button>
                    <div className="text-center mt-4 opacity-20 flex items-center justify-center gap-2">
                        <Package size={8} className="text-white" />
                        <div className="text-[7px] font-black text-white uppercase tracking-[0.4em]">Kitchen AI Standard v1.6.0</div>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
}
