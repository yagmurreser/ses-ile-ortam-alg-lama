"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

const API = "http://localhost:8000";

type Tab = "predict" | "train";

interface PredictResult {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
  chunks_processed: number;
}

interface TrainStatus {
  status: string;
  epoch?: number;
  total_epochs?: number;
  accuracy?: number;
  val_accuracy?: number | null;
  message?: string;
}

interface DatasetStats {
  classes: Record<string, number>;
  total: number;
}

interface AudioFile {
  name: string;
  size: number;
  created_at: number;
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ms: number) {
  return new Date(ms).toLocaleString("tr", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function Spinner({ small }: { small?: boolean }) {
  const size = small ? "w-3.5 h-3.5" : "w-4 h-4";
  return (
    <span
      className={`${size} border-2 border-stone-800 border-t-transparent rounded-full animate-spin inline-block`}
    />
  );
}

function RecordButton({ onRecorded, className }: { onRecorded: (blob: Blob) => void; className?: string }) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    chunksRef.current = [];
    mr.ondataavailable = (e) => chunksRef.current.push(e.data);
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      onRecorded(blob);
      stream.getTracks().forEach((t) => t.stop());
    };
    mr.start();
    mediaRef.current = mr;
    setRecording(true);
    setSeconds(0);
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };

  const stop = () => {
    mediaRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
    setRecording(false);
  };

  return (
    <motion.button
      onClick={recording ? stop : start}
      whileTap={{ scale: 0.97 }}
      className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-colors cursor-pointer ${
        recording
          ? "bg-red-500 text-white"
          : "bg-stone-900 text-white hover:bg-stone-800"
      } ${className ?? ""}`}
    >
      <motion.span
        animate={recording ? { scale: [1, 1.3, 1] } : { scale: 1 }}
        transition={recording ? { repeat: Infinity, duration: 1 } : {}}
        className="w-2 h-2 rounded-full bg-white opacity-80"
      />
      {recording ? `Durdur ${seconds}s` : "Kayıt Başlat"}
    </motion.button>
  );
}

/* ─── PREDICT TAB ─────────────────────────────────────────────────────────── */

function PredictTab() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const predict = useCallback(async (blob: Blob) => {
    setAudioBlob(blob);
    setLoading(true);
    setError("");
    setResult(null);
    const fd = new FormData();
    fd.append("file", blob, "audio.webm");
    try {
      const res = await fetch(`${API}/predict`, { method: "POST", body: fd });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Hata");
      }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Bağlantı hatası");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) predict(f);
    e.target.value = "";
  };

  const isUnknown = result?.label === "bilinmeyen";

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3">
        <RecordButton onRecorded={predict} className="w-full" />
        <label className="flex items-center justify-center px-4 py-2.5 rounded-xl bg-white border border-[var(--border)] hover:bg-[var(--surface-2)] cursor-pointer text-sm font-semibold text-[var(--text)] transition-colors">
          Dosya Seç
          <input type="file" accept="audio/*" className="hidden" onChange={handleFile} />
        </label>
      </div>

      {audioBlob && (
        <div className="bg-white border border-[var(--border)] rounded-xl px-4 py-3">
          <p className="text-xs text-[var(--text-3)] mb-2 font-medium uppercase tracking-wide">
            Yüklenen ses
          </p>
          <audio controls src={URL.createObjectURL(audioBlob)} className="w-full h-9" />
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2.5 text-[var(--text-2)] text-sm">
          <Spinner />
          Analiz ediliyor…
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-red-600 text-sm">
          {error}
        </div>
      )}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="space-y-3"
          >
            <div
              className={`rounded-2xl px-6 py-5 border ${
                isUnknown
                  ? "bg-stone-50 border-[var(--border)]"
                  : "bg-stone-900 border-stone-900"
              }`}
            >
              <p
                className={`text-xs font-medium uppercase tracking-widest mb-1 ${
                  isUnknown ? "text-[var(--text-3)]" : "text-stone-400"
                }`}
              >
                Tahmin
              </p>
              <p
                className={`text-4xl font-bold capitalize tracking-tight ${
                  isUnknown ? "text-[var(--text-2)]" : "text-white"
                }`}
              >
                {result.label}
              </p>
              <div className="flex items-center gap-4 mt-2">
                <span
                  className={`text-sm ${isUnknown ? "text-[var(--text-2)]" : "text-stone-300"}`}
                >
                  Güven:{" "}
                  <span className={`font-semibold ${isUnknown ? "text-[var(--text)]" : "text-white"}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </span>
                <span className={`text-xs ${isUnknown ? "text-[var(--text-3)]" : "text-stone-500"}`}>
                  {result.chunks_processed} chunk
                </span>
              </div>
            </div>

            <div className="bg-white border border-[var(--border)] rounded-2xl px-5 py-4 space-y-3">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--text-3)]">
                Tüm Skorlar
              </p>
              {Object.entries(result.all_scores)
                .sort((a, b) => b[1] - a[1])
                .map(([lbl, score]) => {
                  const isTop = lbl === result.label && !isUnknown;
                  return (
                    <div key={lbl} className="flex items-center gap-3">
                      <span
                        className={`w-20 text-sm capitalize shrink-0 ${
                          isTop ? "text-stone-900 font-semibold" : "text-[var(--text-2)]"
                        }`}
                      >
                        {lbl}
                      </span>
                      <div className="flex-1 bg-stone-100 rounded-full h-1.5 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(score * 100).toFixed(1)}%` }}
                          transition={{ duration: 0.5, ease: "easeOut", delay: 0.1 }}
                          className={`h-full rounded-full ${isTop ? "bg-stone-900" : "bg-stone-300"}`}
                        />
                      </div>
                      <span className="text-sm text-[var(--text-2)] w-12 text-right tabular-nums">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ─── TRAIN TAB ───────────────────────────────────────────────────────────── */

interface UploadJob {
  name: string;
  status: "pending" | "ok" | "error";
  message?: string;
}

function TrainTab() {
  const DEFAULT_CATS = ["avm", "sinif", "dugun"];
  const [categories, setCategories] = useState<string[]>(DEFAULT_CATS);
  const [selectedCat, setSelectedCat] = useState(DEFAULT_CATS[0]);
  const [newCat, setNewCat] = useState("");
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [catFiles, setCatFiles] = useState<AudioFile[]>([]);
  const [catFilesLoading, setCatFilesLoading] = useState(false);
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [trainPolling, setTrainPolling] = useState(false);
  const [epochs, setEpochs] = useState(30);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/dataset/stats`);
      const d: DatasetStats = await res.json();
      setStats(d);
      setCategories((prev) => {
        const merged = [...prev];
        Object.keys(d.classes).forEach((c) => {
          if (!merged.includes(c)) merged.push(c);
        });
        return merged;
      });
    } catch {}
  }, []);

  const fetchCatFiles = useCallback(async (cat: string) => {
    setCatFilesLoading(true);
    try {
      const res = await fetch(`${API}/dataset/files/${cat}`);
      const d = await res.json();
      setCatFiles(d.files ?? []);
    } catch {
      setCatFiles([]);
    } finally {
      setCatFilesLoading(false);
    }
  }, []);

  const fetchTrainStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/train/status`);
      const d: TrainStatus = await res.json();
      setTrainStatus(d);
      return d;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    fetchStats();
    fetchTrainStatus();
  }, [fetchStats, fetchTrainStatus]);

  useEffect(() => {
    fetchCatFiles(selectedCat);
  }, [selectedCat, fetchCatFiles]);

  useEffect(() => {
    if (!trainPolling) return;
    const id = setInterval(async () => {
      const d = await fetchTrainStatus();
      if (!d || d.status === "done" || d.status === "error" || d.status === "idle") {
        setTrainPolling(false);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [trainPolling, fetchTrainStatus]);

  const addCategory = () => {
    const cat = newCat.trim().toLowerCase().replace(/\s+/g, "_");
    if (!cat || categories.includes(cat)) return;
    setCategories((prev) => [...prev, cat]);
    setSelectedCat(cat);
    setNewCat("");
  };

  const uploadOne = async (blob: Blob, filename: string, label: string): Promise<UploadJob> => {
    const fd = new FormData();
    fd.append("label", label);
    fd.append("file", blob, filename);
    try {
      const res = await fetch(`${API}/dataset/upload`, { method: "POST", body: fd });
      const d = await res.json();
      return { name: filename, status: "ok", message: `${d.total_in_class} kayıt` };
    } catch {
      return { name: filename, status: "error", message: "yükleme hatası" };
    }
  };

  const upload = async (blob: Blob) => {
    const ts = Date.now();
    const job = await uploadOne(blob, `kayit_${ts}.webm`, selectedCat);
    setJobs([job]);
    fetchStats();
    fetchCatFiles(selectedCat);
  };

  const handleFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setJobs(files.map((f) => ({ name: f.name, status: "pending" })));
    for (let i = 0; i < files.length; i++) {
      const result = await uploadOne(files[i], files[i].name, selectedCat);
      setJobs((prev) => prev.map((j, idx) => (idx === i ? result : j)));
    }
    fetchStats();
    fetchCatFiles(selectedCat);
    e.target.value = "";
  };

  const startTrain = async () => {
    await fetch(`${API}/train?epochs=${epochs}`, { method: "POST" });
    setTrainPolling(true);
    fetchTrainStatus();
  };

  const hasData = (stats?.total ?? 0) > 0;
  const trainProgress =
    trainStatus?.epoch && trainStatus?.total_epochs
      ? Math.round((trainStatus.epoch / trainStatus.total_epochs) * 100)
      : 0;

  return (
    <div className="space-y-6">
      {/* Category selector */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--text-3)]">
          Kategoriler
        </p>
        <div className="flex flex-wrap gap-2">
          {categories.map((cat) => (
            <motion.button
              key={cat}
              onClick={() => setSelectedCat(cat)}
              whileTap={{ scale: 0.96 }}
              className={`relative px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                selectedCat === cat
                  ? "bg-stone-900 text-white"
                  : "bg-white border border-[var(--border)] text-[var(--text-2)] hover:border-stone-400 hover:text-[var(--text)]"
              }`}
            >
              {cat}
              {stats?.classes[cat] !== undefined && (
                <span
                  className={`ml-1.5 text-xs ${
                    selectedCat === cat ? "opacity-50" : "text-[var(--text-3)]"
                  }`}
                >
                  {stats.classes[cat]}
                </span>
              )}
            </motion.button>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            value={newCat}
            onChange={(e) => setNewCat(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addCategory()}
            placeholder="yeni kategori…"
            className="bg-white border border-[var(--border)] rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-stone-400 text-[var(--text)] placeholder:text-[var(--text-3)] transition-colors"
          />
          <button
            onClick={addCategory}
            disabled={!newCat.trim()}
            className="px-4 py-2 rounded-lg bg-white border border-[var(--border)] text-sm font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-stone-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
          >
            + Ekle
          </button>
        </div>
      </div>

      {/* Selected category files */}
      <div className="bg-white border border-[var(--border)] rounded-2xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[var(--border)] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-[var(--text)] capitalize">{selectedCat}</span>
            {!catFilesLoading && (
              <span className="text-xs text-[var(--text-3)] font-medium">{catFiles.length} dosya</span>
            )}
          </div>
          {catFilesLoading && <Spinner small />}
        </div>

        <AnimatePresence mode="wait">
          {catFiles.length === 0 && !catFilesLoading ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="px-5 py-8 text-center"
            >
              <p className="text-sm text-[var(--text-3)]">Bu kategoride henüz ses yok</p>
            </motion.div>
          ) : (
            <motion.div
              key={selectedCat}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="divide-y divide-[var(--border)] max-h-64 overflow-y-auto"
            >
              {catFiles.map((f, i) => (
                <motion.div
                  key={f.name}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.15, delay: i * 0.03 }}
                  className="flex items-center gap-3 px-5 py-3 hover:bg-[var(--surface-2)] transition-colors"
                >
                  <span className="text-xs text-[var(--text-3)] w-5 shrink-0 text-right tabular-nums">
                    {i + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-[var(--text)] truncate font-medium">{f.name}</p>
                    <p className="text-xs text-[var(--text-3)]">
                      {formatSize(f.size)} · {formatDate(f.created_at)}
                    </p>
                  </div>
                  <audio
                    controls
                    src={`${API}/dataset/audio/${selectedCat}/${f.name}`}
                    className="h-7 shrink-0"
                    style={{ width: 140 }}
                  />
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Upload actions */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--text-3)] mb-3">
          Ses Ekle
        </p>
        <div className="flex flex-wrap gap-3 items-center">
          <RecordButton onRecorded={upload} />
          <label className="px-4 py-2.5 rounded-xl bg-white border border-[var(--border)] hover:bg-[var(--surface-2)] cursor-pointer text-sm font-semibold text-[var(--text)] transition-colors">
            Dosya Yükle
            <input type="file" accept="audio/*" multiple className="hidden" onChange={handleFiles} />
          </label>
        </div>
      </div>

      {/* Job list */}
      <AnimatePresence>
        {jobs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="bg-white border border-[var(--border)] rounded-xl px-4 py-3 space-y-1.5 max-h-40 overflow-y-auto"
          >
            {jobs.map((j, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                {j.status === "pending" ? (
                  <Spinner small />
                ) : (
                  <span
                    className={`w-2 h-2 rounded-full shrink-0 ${
                      j.status === "ok" ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                )}
                <span className="text-[var(--text)] truncate flex-1">{j.name}</span>
                {j.message && (
                  <span className="text-[var(--text-3)] text-xs shrink-0">{j.message}</span>
                )}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Train section */}
      {hasData && (
        <div className="border-t border-[var(--border)] pt-6 space-y-4">
          <div>
            <p className="text-sm font-semibold text-[var(--text)]">Model Eğit</p>
            <p className="text-xs text-[var(--text-3)] mt-0.5">
              Toplam{" "}
              <span className="font-medium text-[var(--text-2)]">{stats?.total}</span> ses dosyası
            </p>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2 bg-white border border-[var(--border)] rounded-lg px-3 py-2">
              <label className="text-xs font-medium text-[var(--text-3)] whitespace-nowrap">
                Epoch
              </label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Math.max(1, Number(e.target.value)))}
                min={1}
                max={500}
                className="w-16 text-sm text-[var(--text)] bg-transparent focus:outline-none text-center font-semibold"
              />
            </div>
            <motion.button
              onClick={startTrain}
              disabled={trainStatus?.status === "running"}
              whileTap={{ scale: 0.97 }}
              className="px-5 py-2.5 rounded-xl bg-stone-900 hover:bg-stone-800 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors cursor-pointer"
            >
              {trainStatus?.status === "running" ? "Eğitiliyor…" : "Eğit"}
            </motion.button>
          </div>

          <AnimatePresence>
            {trainStatus && trainStatus.status !== "idle" && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2 }}
                className="bg-white border border-[var(--border)] rounded-xl px-5 py-4 space-y-3"
              >
                {trainStatus.status === "running" && (
                  <>
                    <div className="flex items-center justify-between text-xs text-[var(--text-2)]">
                      <span className="flex items-center gap-2">
                        <Spinner small />
                        Epoch {trainStatus.epoch} / {trainStatus.total_epochs}
                      </span>
                      <span className="flex gap-3 font-semibold tabular-nums">
                        {trainStatus.accuracy !== undefined && (
                          <span>Train {(trainStatus.accuracy * 100).toFixed(1)}%</span>
                        )}
                        {trainStatus.val_accuracy != null && (
                          <span className="text-stone-900">
                            Val {(trainStatus.val_accuracy * 100).toFixed(1)}%
                          </span>
                        )}
                      </span>
                    </div>
                    <div className="bg-stone-100 rounded-full h-1.5 overflow-hidden">
                      <motion.div
                        className="bg-stone-900 h-full rounded-full"
                        animate={{ width: `${trainProgress}%` }}
                        transition={{ duration: 0.5, ease: "easeOut" }}
                      />
                    </div>
                  </>
                )}
                {trainStatus.status === "done" && (
                  <p className="text-sm text-green-700 font-medium">
                    Tamamlandı —{" "}
                    {trainStatus.accuracy !== undefined &&
                      `train: ${(trainStatus.accuracy * 100).toFixed(1)}%`}
                    {trainStatus.val_accuracy != null &&
                      ` · val: ${(trainStatus.val_accuracy * 100).toFixed(1)}%`}
                  </p>
                )}
                {trainStatus.status === "error" && (
                  <p className="text-sm text-red-600">{trainStatus.message}</p>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

/* ─── ROOT ────────────────────────────────────────────────────────────────── */

export default function Home() {
  const [tab, setTab] = useState<Tab>("predict");
  const [modelInfo, setModelInfo] = useState<{
    loaded: boolean;
    labels?: string[];
    num_classes?: number;
  } | null>(null);

  useEffect(() => {
    fetch(`${API}/model/info`)
      .then((r) => r.json())
      .then(setModelInfo)
      .catch(() => setModelInfo({ loaded: false }));
  }, []);

  const tabs: { id: Tab; label: string }[] = [
    { id: "predict", label: "Tahmin" },
    { id: "train", label: "Eğitme" },
  ];

  return (
    <main className="min-h-screen p-6 w-full max-w-[680px] mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-2xl font-bold text-[var(--text)] tracking-tight">
          Ses Sınıflandırma
        </h1>
        {modelInfo && (
          <p className="text-sm text-[var(--text-3)] mt-1">
            {modelInfo.loaded ? (
              <>
                Model yüklü —{" "}
                <span className="text-[var(--text-2)]">
                  {modelInfo.num_classes} sınıf: {modelInfo.labels?.join(", ")}
                </span>
              </>
            ) : (
              "Model yüklü değil — önce veri topla ve eğit"
            )}
          </p>
        )}
      </div>

      {/* Modern underline tabs */}
      <div className="grid grid-cols-2 border-b border-[var(--border)] mb-6">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className="relative pb-3 text-sm font-medium cursor-pointer transition-colors text-center"
          >
            <span
              className={
                tab === t.id ? "text-[var(--text)]" : "text-[var(--text-3)] hover:text-[var(--text-2)]"
              }
            >
              {t.label}
            </span>
            {tab === t.id && (
              <motion.div
                layoutId="tab-underline"
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-stone-900 rounded-full"
                transition={{ type: "spring", stiffness: 500, damping: 35 }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Card with animated content switch */}
      <div className="bg-white border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            {tab === "predict" && <PredictTab />}
            {tab === "train" && <TrainTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}
