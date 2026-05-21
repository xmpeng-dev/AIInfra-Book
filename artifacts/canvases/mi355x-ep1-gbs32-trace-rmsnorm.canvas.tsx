/**
 * MI355X single-GPU trace report - TP1 PP1 EP=1, GBS=32, MBS=4.
 * Run with the RMSNorm sustained optimization fully enabled (Triton
 * RMSNorm at TENorm sites + split LayerNormColumnParallelLinear that
 * routes the linear_qkv norm through the same Triton kernel).
 *
 * Source:  gpt-oss-20B FP8 hybrid, rank 2, ProfilerStep #17
 * Trace:   small_llm_moe_pretraining/primus/run-trace/
 *          20260425_b_rmsnorm_trace/output/.../rank[2]....pt.trace.json
 * Numbers: .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py
 *
 * Canonical location for this file is `slab/canvases/`. A symlink at
 * `~/.cursor/projects/home-xiaompen-mlperf-training/canvases/` points
 * here so the IDE still renders it.
 */
import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/* ── Run identity ─────────────────────────────────────────────────────── */
const RUN = {
  hardware: "8 × MI355X",
  model: "gpt-oss-20B FP8 hybrid",
  parallelism: "TP1 PP1 EP1",
  batch: "GBS 32 · MBS 4",
  step: "ProfilerStep #17",
  stepMs: 1127.56,
  samplesPerStep: 32,
  activeStreams: 8,
  computeBusyPct: 95.3,
  ncclHiddenPct: 81.6,
  streamOversubPct: 151,
  // For perspective vs the previous canvas (no LayerNormLinear split, before
  // 2026-04-25). Same hardware, same yaml, same env; only the
  // rms_norm_patches.py site-2 patch differs.
  prevStepMs: 1129.33,
};

const TFLOPS_PER_GPU = 691.4;     // from run.log iter 20 line, current iter
const PREV_TFLOPS_PER_GPU = 690.0; // mi355x-ep1-gbs32-trace.canvas.tsx baseline

/* ── Per-kernel-category busy time across all streams (ms) ────────────── */
type KernelRow = {
  category: string;
  ms: number;
  prevMs?: number;          // previous canvas value, when comparable
  tone?: "info" | "warning";
};
const KERNEL_ROWS: KernelRow[] = [
  { category: "GEMM (dense + grouped MoE)", ms: 933.55, prevMs: 941.07, tone: "info" },
  { category: "Attention (FMHA fwd/bwd)", ms: 203.66, prevMs: 203.21 },
  { category: "Elementwise / cast / activation", ms: 159.25, prevMs: 160.28 },
  { category: "NCCL all-to-all (MoE-style)", ms: 128.20, prevMs: 129.50, tone: "warning" },
  { category: "NCCL AllGather (DDP)", ms: 98.54, prevMs: 98.36, tone: "warning" },
  { category: "Other (autograd / misc)", ms: 53.17, prevMs: 53.10 },
  { category: "RMSNorm (Triton, all sites)", ms: 39.10, prevMs: 39.22 },
  { category: "MoE dispatch / permute / topk", ms: 27.98, prevMs: 28.08 },
  { category: "Optimizer / param update", ms: 16.63, prevMs: 16.63 },
  { category: "Reduction (sum / mean)", ms: 15.93, prevMs: 16.01 },
  { category: "MemCopy / alloc", ms: 11.34, prevMs: 11.52 },
  { category: "NCCL ReduceScatter (DDP)", ms: 12.05, prevMs: 9.35, tone: "warning" },
  { category: "Softmax", ms: 2.92 },
  { category: "FP8 cast", ms: 0.04 },
];

const STREAM_ROWS: Array<{
  stream: string;
  role: string;
  busyMs: number;
  share: string;
}> = [
  { stream: "stream 0", role: "Compute (attn · GEMM · norm · MoE · opt)", busyMs: 603, share: "53.5%" },
  { stream: "stream 11", role: "RCCL DDP (RS + AG + AR, dedicated comm)", busyMs: 238, share: "21.1%" },
  { stream: "stream 14", role: "Parallel grouped GEMM (expert lane)", busyMs: 220, share: "19.5%" },
  { stream: "stream 13", role: "Parallel grouped GEMM (expert lane)", busyMs: 205, share: "18.2%" },
  { stream: "stream 16", role: "Parallel grouped GEMM (expert lane)", busyMs: 201, share: "17.8%" },
  { stream: "stream 15", role: "Parallel grouped GEMM (expert lane)", busyMs: 181, share: "16.0%" },
  { stream: "stream 4", role: "MemCopy / D2D", busyMs: 55, share: "4.9%" },
  { stream: "stream 12", role: "(idle, negligible)", busyMs: 0, share: "0.0%" },
];

/* ── RMSNorm patch evidence (from run.log + top-25 GPU kernel list) ──── */
const RMSNORM_SITES: Array<{ site: string; routedTo: string; via: string }> = [
  { site: "linear_qkv (fused norm + linear)", routedTo: "PrimusTurboLayerNormColumnParallelLinear", via: "spec_provider patch (site 2)" },
  { site: "q_layernorm", routedTo: "PrimusTurboRMSNorm", via: "te.pytorch.RMSNorm patch (site 1)" },
  { site: "k_layernorm", routedTo: "PrimusTurboRMSNorm", via: "te.pytorch.RMSNorm patch (site 1)" },
  { site: "pre_mlp_layernorm", routedTo: "PrimusTurboRMSNorm", via: "te.pytorch.RMSNorm patch (site 1)" },
  { site: "final_layernorm", routedTo: "PrimusTurboRMSNorm", via: "te.pytorch.RMSNorm patch (site 1)" },
];

/* ── Top-5 GPU kernels (raw names cleaned) ───────────────────────────── */
const TOP_KERNELS: Array<{ name: string; ms: number; pct: number; note?: string }> = [
  { name: "ncclDevKernel_Generic_1 (RCCL)", ms: 238.78, pct: 21.2 },
  { name: "Cijk MT192x192x64 (grouped MoE GEMM, fwd)", ms: 237.72, pct: 21.1, note: "stream 14/13/16/15" },
  { name: "Cijk MT256x256x64 BBS (grouped MoE GEMM, bwd dW)", ms: 237.10, pct: 21.0, note: "stream 14/13/16/15" },
  { name: "Cijk MT256x256x64 (grouped MoE GEMM, bwd dX)", ms: 185.49, pct: 16.5, note: "stream 14/13/16/15" },
  { name: "aiter::fmha_bwd_hd64_bf16_causal_a16 (FMHA bwd v3)", ms: 125.58, pct: 11.1, note: "stream 0" },
  { name: "_rmsnorm_bwd_kernel (Triton, this patch)", ms: 26.28, pct: 2.3, note: "evidence the Triton path is live" },
];

/* ── Multi-stream pipeline lanes for one ProfilerStep ─────────────────── */
type SegKind =
  | "gemm" | "attn" | "norm" | "moe" | "elem" | "opt"
  | "a2a" | "rs" | "ag" | "ar" | "idle";
type Seg = { kind: SegKind; t: number; w: number; label?: string };

// Step shape (from time-binned mix, bin = 14.1 ms):
//   bins  0–22  (0–310 ms)   forward   ~12 ms gemm + 3 ms attn + 4 ms NCCL/bin
//   bins 23–77 (325–1090 ms) backward  ~16-22 ms gemm + 7-11 ms FMHA bwd + 5 ms NCCL/bin
//   bins 78–79 (1099–1127 ms) optimizer / step tail (largely "other" + memcpy)
const LANES: Array<{ name: string; tag: string; segs: Seg[] }> = [
  {
    name: "stream 0", tag: "compute (attn + dense GEMM + norm)",
    segs: [
      { kind: "elem", t: 0, w: 1 },
      { kind: "norm", t: 1, w: 1 },
      { kind: "attn", t: 2, w: 4, label: "fmha fwd" },
      { kind: "gemm", t: 6, w: 4, label: "qkv / proj" },
      { kind: "norm", t: 10, w: 1 },
      { kind: "moe", t: 11, w: 1 },
      { kind: "gemm", t: 12, w: 4 },
      { kind: "attn", t: 16, w: 4, label: "fmha fwd" },
      { kind: "gemm", t: 20, w: 6, label: "fwd dense" },
      { kind: "norm", t: 26, w: 1 },
      { kind: "elem", t: 27, w: 1 },
      // backward starts (~325 ms = bin 23, ≈ 29% of step)
      { kind: "gemm", t: 28, w: 6, label: "bwd dense" },
      { kind: "attn", t: 34, w: 6, label: "fmha bwd" },
      { kind: "gemm", t: 40, w: 6 },
      { kind: "elem", t: 46, w: 2 },
      { kind: "attn", t: 48, w: 6, label: "fmha bwd" },
      { kind: "gemm", t: 54, w: 6 },
      { kind: "norm", t: 60, w: 1 },
      { kind: "elem", t: 61, w: 2 },
      { kind: "attn", t: 63, w: 6, label: "fmha bwd" },
      { kind: "gemm", t: 69, w: 6 },
      { kind: "attn", t: 75, w: 5, label: "fmha bwd" },
      { kind: "gemm", t: 80, w: 6 },
      { kind: "norm", t: 86, w: 1 },
      { kind: "elem", t: 87, w: 1 },
      { kind: "attn", t: 88, w: 5, label: "fmha bwd" },
      { kind: "gemm", t: 93, w: 4 },
      // optimizer tail (~1099-1127 ms = bins 78-79, last ~2.5%)
      { kind: "opt", t: 97, w: 3, label: "step" },
    ],
  },
  {
    name: "stream 14", tag: "grouped GEMM lane",
    segs: [
      { kind: "idle", t: 0, w: 1 },
      { kind: "gemm", t: 1, w: 26, label: "fwd grouped MoE" },
      { kind: "idle", t: 27, w: 2 },
      { kind: "gemm", t: 29, w: 35, label: "bwd grouped MoE" },
      { kind: "idle", t: 64, w: 36 },
    ],
  },
  {
    name: "stream 13", tag: "grouped GEMM lane",
    segs: [
      { kind: "idle", t: 0, w: 2 },
      { kind: "gemm", t: 2, w: 24 },
      { kind: "idle", t: 26, w: 3 },
      { kind: "gemm", t: 29, w: 32 },
      { kind: "idle", t: 61, w: 39 },
    ],
  },
  {
    name: "stream 16", tag: "grouped GEMM lane",
    segs: [
      { kind: "idle", t: 0, w: 2 },
      { kind: "gemm", t: 2, w: 24 },
      { kind: "idle", t: 26, w: 3 },
      { kind: "gemm", t: 29, w: 30 },
      { kind: "idle", t: 59, w: 41 },
    ],
  },
  {
    name: "stream 15", tag: "grouped GEMM lane",
    segs: [
      { kind: "idle", t: 0, w: 3 },
      { kind: "gemm", t: 3, w: 22 },
      { kind: "idle", t: 25, w: 4 },
      { kind: "gemm", t: 29, w: 28 },
      { kind: "idle", t: 57, w: 43 },
    ],
  },
  {
    name: "stream 11", tag: "RCCL DDP (overlapped)",
    segs: [
      { kind: "idle", t: 0, w: 1 },
      { kind: "rs", t: 1, w: 3, label: "rs (param 0)" },
      { kind: "ag", t: 4, w: 4, label: "ag" },
      { kind: "rs", t: 8, w: 3 },
      { kind: "ag", t: 11, w: 4 },
      { kind: "rs", t: 15, w: 3 },
      { kind: "ag", t: 18, w: 4 },
      { kind: "rs", t: 22, w: 3 },
      { kind: "ag", t: 25, w: 3 },
      { kind: "rs", t: 30, w: 3, label: "rs (grad bucket 0)" },
      { kind: "ag", t: 33, w: 4 },
      { kind: "rs", t: 38, w: 3 },
      { kind: "ag", t: 41, w: 4 },
      { kind: "rs", t: 46, w: 3 },
      { kind: "ag", t: 49, w: 3 },
      { kind: "rs", t: 53, w: 3 },
      { kind: "ag", t: 56, w: 3 },
      { kind: "rs", t: 60, w: 3 },
      { kind: "ag", t: 63, w: 3 },
      { kind: "rs", t: 67, w: 3 },
      { kind: "ag", t: 70, w: 3 },
      { kind: "rs", t: 74, w: 3 },
      { kind: "ag", t: 77, w: 3 },
      { kind: "rs", t: 81, w: 3 },
      { kind: "ag", t: 84, w: 3 },
      { kind: "ar", t: 88, w: 1, label: "ar" },
      { kind: "rs", t: 89, w: 3 },
      { kind: "ag", t: 92, w: 3 },
      { kind: "ar", t: 96, w: 1 },
      { kind: "idle", t: 97, w: 3 },
    ],
  },
  {
    name: "stream 4", tag: "MemCopy / D2D",
    segs: [
      { kind: "elem", t: 0, w: 1, label: "h2d" },
      { kind: "idle", t: 1, w: 22 },
      { kind: "elem", t: 23, w: 5, label: "d2d (bwd)" },
      { kind: "idle", t: 28, w: 50 },
      { kind: "elem", t: 78, w: 1 },
      { kind: "idle", t: 79, w: 21 },
    ],
  },
];

/* ── palette ──────────────────────────────────────────────────────────── */
const SEG_COLORS: Record<SegKind, string> = {
  gemm: "#2E79B5E0",
  attn: "#7B64B8F0",
  norm: "#70B0D8E0",
  moe: "#F0A040E0",
  elem: "#8888A8E0",
  opt: "#1F8A65E8",
  a2a: "#C04848E0",
  rs: "#E8C030E0",
  ag: "#7DCAB0E0",
  ar: "#C85898E0",
  idle: "transparent",
};

function useSegPalette() {
  const t = useHostTheme();
  return { ...SEG_COLORS, idle: t.fill.tertiary } as Record<SegKind, string>;
}

function PipelineDiagram({
  title, totalMs, lanes,
}: { title: string; totalMs: number; lanes: typeof LANES }) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const labelW = 160;
  const trackH = 22;
  const gap = 6;
  const width = 760;
  const trackW = width - labelW;
  const height = lanes.length * (trackH + gap) + 28;

  return (
    <Stack gap={8}>
      <Row align="center" gap={8}>
        <Text weight="semibold">{title}</Text>
        <Text tone="secondary" size="small">
          one ProfilerStep ≈ {totalMs.toFixed(0)} ms · time →
        </Text>
      </Row>
      <svg width={width} height={height} style={{ display: "block" }}>
        {[0, 25, 50, 75, 100].map((p) => {
          const x = labelW + (trackW * p) / 100;
          return (
            <g key={p}>
              <line x1={x} x2={x} y1={0}
                y2={lanes.length * (trackH + gap)}
                stroke={t.stroke.tertiary} strokeDasharray="2 3" />
              <text x={x} y={lanes.length * (trackH + gap) + 16}
                fontSize={10} fill={t.text.quaternary} textAnchor="middle">
                {((p / 100) * totalMs).toFixed(0)} ms
              </text>
            </g>
          );
        })}
        {lanes.map((lane, i) => {
          const y = i * (trackH + gap);
          return (
            <g key={lane.name}>
              <text x={0} y={y + trackH / 2 + 4} fontSize={11}
                fill={t.text.secondary}
                fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace">
                {lane.name}
              </text>
              <text x={80} y={y + trackH / 2 + 4} fontSize={10} fill={t.text.quaternary}>
                {lane.tag}
              </text>
              <rect x={labelW} y={y} width={trackW} height={trackH}
                fill={t.fill.quaternary} rx={3} />
              {lane.segs.map((s, j) => {
                const sx = labelW + (trackW * s.t) / 100;
                const sw = (trackW * s.w) / 100;
                const isIdle = s.kind === "idle";
                return (
                  <g key={j}>
                    <rect x={sx} y={y} width={sw} height={trackH}
                      fill={isIdle ? "transparent" : palette[s.kind]}
                      opacity={isIdle ? 0 : 0.92} rx={2}>
                      <title>{`${s.kind}${s.label ? " · " + s.label : ""} (${s.w}%)`}</title>
                    </rect>
                    {s.label && sw > 38 ? (
                      <text x={sx + 4} y={y + trackH / 2 + 3} fontSize={9}
                        fill={t.text.onAccent ?? "#fff"}
                        style={{ pointerEvents: "none" }}>
                        {s.label}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </Stack>
  );
}

function Legend() {
  const palette = useSegPalette();
  const items: Array<{ kind: SegKind; label: string }> = [
    { kind: "gemm", label: "GEMM (dense + grouped)" },
    { kind: "attn", label: "Attention (FMHA)" },
    { kind: "norm", label: "Norm" },
    { kind: "moe", label: "MoE permute / topk" },
    { kind: "elem", label: "Elementwise / cast" },
    { kind: "opt", label: "Optimizer" },
    { kind: "rs", label: "RCCL ReduceScatter" },
    { kind: "ag", label: "RCCL AllGather" },
    { kind: "ar", label: "RCCL AllReduce" },
  ];
  return (
    <Row gap={12} wrap>
      {items.map((it) => (
        <Row key={it.kind} gap={6} align="center">
          <span style={{
            display: "inline-block", width: 12, height: 12,
            background: palette[it.kind], borderRadius: 2,
          }} />
          <Text size="small" tone="secondary">{it.label}</Text>
        </Row>
      ))}
    </Row>
  );
}

const fmt = (n: number, d = 1) => n.toFixed(d);
const fmtDelta = (n: number, d = 1) => (n >= 0 ? "+" : "") + n.toFixed(d);

const NCCL_TOTAL =
  KERNEL_ROWS.find((r) => r.category.startsWith("NCCL all-to-all"))!.ms +
  KERNEL_ROWS.find((r) => r.category.startsWith("NCCL AllGather"))!.ms +
  KERNEL_ROWS.find((r) => r.category.startsWith("NCCL ReduceScatter"))!.ms;

const COMPUTE_TOTAL =
  KERNEL_ROWS.filter((r) => !r.category.startsWith("NCCL"))
    .reduce((s, r) => s + r.ms, 0);

export default function MI355XEp1Gbs32TraceRmsnorm() {
  const msPerSample = RUN.stepMs / RUN.samplesPerStep;
  const stepDelta = RUN.stepMs - RUN.prevStepMs;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>MI355X · single-GPU trace with RMSNorm sustained patch</H1>
        <Row gap={8} wrap>
          <Pill tone="info">{RUN.model}</Pill>
          <Pill tone="neutral">{RUN.parallelism}</Pill>
          <Pill tone="neutral">{RUN.batch}</Pill>
          <Pill tone="neutral">{RUN.step}</Pill>
          <Pill tone="success">PrimusTurboRMSNorm at all 5 norm sites</Pill>
        </Row>
      </Stack>

      <Callout tone="info" title="What's new in this trace">
        Both halves of the RMSNorm patch are live now: <Text as="span" weight="semibold">
        te.pytorch.RMSNorm</Text> is swapped to{" "}
        <Text as="span" weight="semibold">PrimusTurboRMSNorm</Text> (Triton),{" "}
        <Text as="span">and</Text> the spec provider's{" "}
        <Text as="span" weight="semibold">TELayerNormColumnParallelLinear</Text>{" "}
        is swapped to <Text as="span" weight="semibold">
        PrimusTurboLayerNormColumnParallelLinear</Text> so the norm fused
        inside <Text as="span">linear_qkv</Text> also goes through the same
        Triton kernel. Trace evidence: <Text as="span">_rmsnorm_bwd_kernel</Text>{" "}
        appears in the top-25 GPU kernels at 26.3 ms (2.3% of step), and the
        run.log shows all 5 RMSNorm sites bound to the new classes.
      </Callout>

      <Grid columns={5} gap={12}>
        <Stat value={`${fmt(RUN.stepMs, 0)} ms`} label="step time" tone="success" />
        <Stat value={`${fmt(msPerSample, 1)} ms`} label="per sample" tone="success" />
        <Stat value={`${fmt(TFLOPS_PER_GPU, 0)}`} label="TFLOP/s/GPU (iter)" tone="success" />
        <Stat value={`${fmt(RUN.computeBusyPct, 0)}%`} label="compute-stream busy" tone="success" />
        <Stat value={`${fmt(RUN.ncclHiddenPct, 0)}%`} label="NCCL hidden behind compute" tone="success" />
      </Grid>

      <Grid columns={3} gap={12}>
        <Stat value={`${RUN.streamOversubPct}%`} label="stream oversubscription" />
        <Stat value={`${fmtDelta(stepDelta, 1)} ms`} label="step Δ vs prev canvas" />
        <Stat value={`${fmtDelta(TFLOPS_PER_GPU - PREV_TFLOPS_PER_GPU, 1)}`} label="TFLOP/s/GPU Δ vs prev" />
      </Grid>

      <Divider />

      <H2>Per-stream timeline (one ProfilerStep)</H2>
      <Text tone="secondary">
        Each lane is a HIP stream; segments are colored by kernel category.
        Phase A (0-310 ms ≈ 27%) is the forward pass; phase B (310-1090 ms ≈
        69%) is backward; the last 40 ms (≈ 3.5%) is the optimizer step.
        DDP collectives on stream 11 are emitted in flight with the
        backward GEMMs, which is why they don&apos;t add to the wall.
      </Text>
      <Legend />

      <Card>
        <CardHeader trailing={<Pill tone="success">{RUN.activeStreams} active streams</Pill>}>
          MI355X · 1 compute lane + 4 grouped-GEMM lanes + 1 RCCL lane
        </CardHeader>
        <CardBody>
          <PipelineDiagram
            title="Compute on stream 0; experts shard across 4 grouped-GEMM streams; DDP RS/AG runs in parallel."
            totalMs={RUN.stepMs}
            lanes={LANES}
          />
        </CardBody>
      </Card>

      <Divider />

      <H2>RMSNorm patch sites (run.log evidence)</H2>
      <Text tone="secondary">
        Five norm sites in the model. All resolve to the Triton-backed
        classes after the rms_norm_patches.py runtime patch. Site 1 is the
        original patch (te.pytorch.RMSNorm swap, covers all standalone
        norms); site 2 is the new spec_provider swap that catches the norm
        baked into the fused linear_qkv module.
      </Text>
      <Table
        headers={["Module site", "Resolved class", "Patched via"]}
        columnAlign={["left", "left", "left"]}
        rows={RMSNORM_SITES.map((r) => [r.site, r.routedTo, r.via])}
      />

      <Divider />

      <H2>GPU work decomposition</H2>
      <Text tone="secondary">
        Sum of busy time per kernel category across all streams (ms). Numbers
        can exceed the {fmt(RUN.stepMs, 0)} ms wall because the four
        grouped-GEMM lanes and the RCCL lane run concurrently with stream 0.
        Δ column is vs the previous canvas (same yaml, same env, only
        rms_norm_patches.py site 2 differs).
      </Text>

      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>Kernel breakdown</H3>
          <Table
            headers={["Category", "Busy (ms)", "% of step", "Δ (ms)"]}
            columnAlign={["left", "right", "right", "right"]}
            rows={KERNEL_ROWS.map((r) => [
              r.category, fmt(r.ms, 1),
              `${fmt((r.ms / RUN.stepMs) * 100, 1)}%`,
              r.prevMs == null ? "—" : fmtDelta(r.ms - r.prevMs, 1),
            ])}
            rowTone={KERNEL_ROWS.map((r) => r.tone)}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Compute vs communication share</H3>
          <BarChart
            categories={["MI355X step"]}
            stacked normalized height={360} valueSuffix="%"
            series={[
              { name: "GEMM", tone: "info",
                data: [KERNEL_ROWS.find((r) => r.category.startsWith("GEMM"))!.ms] },
              { name: "Attention",
                data: [KERNEL_ROWS.find((r) => r.category.startsWith("Attention"))!.ms] },
              { name: "Norm + Elem + Opt + Mem + Reduction + MoE", tone: "neutral",
                data: [
                  KERNEL_ROWS.filter((r) =>
                    /^(Elementwise|Other|RMSNorm|MoE|Optimizer|Reduction|MemCopy|Softmax|FP8)/.test(r.category)
                  ).reduce((s, r) => s + r.ms, 0),
                ] },
              { name: "RCCL DDP (RS + AG)", tone: "warning",
                data: [
                  KERNEL_ROWS.find((r) => r.category.startsWith("NCCL ReduceScatter"))!.ms +
                  KERNEL_ROWS.find((r) => r.category.startsWith("NCCL AllGather"))!.ms,
                ] },
              { name: "RCCL all-to-all", tone: "danger",
                data: [KERNEL_ROWS.find((r) => r.category.startsWith("NCCL all-to-all"))!.ms] },
            ]}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>Top GPU kernels</H2>
      <Table
        headers={["Kernel", "Busy (ms)", "% of step", "Where / note"]}
        columnAlign={["left", "right", "right", "left"]}
        rows={TOP_KERNELS.map((k) => [k.name, fmt(k.ms, 1), `${fmt(k.pct, 1)}%`, k.note ?? "—"])}
      />

      <Divider />

      <H2>Stream occupancy</H2>
      <Card>
        <CardHeader trailing={<Pill size="sm">{RUN.activeStreams} active streams</Pill>}>
          Per-stream busy time
        </CardHeader>
        <CardBody>
          <Table
            headers={["Stream", "Role", "Busy (ms)", "Share of step"]}
            columnAlign={["left", "left", "right", "right"]}
            rows={STREAM_ROWS.map((r) => [
              r.stream, r.role, fmt(r.busyMs, 0), r.share,
            ])}
          />
        </CardBody>
      </Card>

      <Grid columns={4} gap={12}>
        <Stat value={`${fmt(NCCL_TOTAL, 0)} ms`}
          label="total RCCL kernel time" tone="warning" />
        <Stat value={`${fmt(COMPUTE_TOTAL, 0)} ms`}
          label="total compute kernel time" tone="info" />
        <Stat value={`${fmt(NCCL_TOTAL / RUN.stepMs * 100, 0)}%`}
          label="RCCL as % of step (raw)" />
        <Stat value={`${fmt(RUN.ncclHiddenPct, 0)}%`}
          label="of which hidden behind compute" tone="success" />
      </Grid>

      <Divider />

      <H2>Reading the timeline</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>What this trace confirms</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                <Text as="span" weight="semibold">All 5 RMSNorm sites resolve
                to the Triton kernel.</Text>{" "}
                run.log emits both patch lines on rank 0; named_modules dump
                shows linear_qkv as PrimusTurboLayerNormColumnParallelLinear
                (norm=PrimusTurboRMSNorm) and the four standalone norm sites
                as PrimusTurboRMSNorm. The trace confirms with{" "}
                <Text as="span">_rmsnorm_bwd_kernel</Text> in the top-25.
              </Text>
              <Text>
                <Text as="span" weight="semibold">Legacy GroupedMLP still wins at EP=1.</Text>{" "}
                The four grouped-GEMM streams (13/14/15/16) carry 180-220 ms
                of expert GEMM in parallel with stream 0, exactly as in the
                pre-rmsnorm canvas. use_turbo_grouped_mlp is pinned to
                literal false in yaml; the env-var truthy fix from 04-25 is
                holding.
              </Text>
              <Text>
                <Text as="span" weight="semibold">DDP collectives still 81.6% hidden.</Text>{" "}
                stream 11 runs 238 ms of RS/AG/AR but 198 ms of that lands
                under compute on stream 0. Net exposed comms ≈ 45 ms, which
                is dwarfed by the 934 ms of GEMM.
              </Text>
              <Text>
                <Text as="span" weight="semibold">Stream oversubscription 151%.</Text>{" "}
                Sum of per-stream busy time is 1702 ms vs a 1128 ms wall.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>What did the rmsnorm patch actually buy here?</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                <Text as="span" weight="semibold">At GBS=32 EP=1 the marginal
                step-time gain is small.</Text>{" "}
                Step wall went from 1129.3 ms to 1127.6 ms (-1.7 ms,
                -0.15%). RMSNorm category itself moved 39.22 → 39.10 ms.
                Total GEMM moved 941 → 934 ms (most of the 7 ms delta is
                noise between runs of the same class).
              </Text>
              <Text>
                <Text as="span" weight="semibold">Why the visible gain is small here.</Text>{" "}
                The previous canvas already had site 1 of the patch (te.pytorch
                RMSNorm swap) so the q/k/pre_mlp/final norms were already
                Triton. Site 2 only swaps the linear_qkv-fused norm; on this
                workload that sits on stream 0 mostly underneath GEMM, so
                shaving its kernel time doesn&apos;t move the wall by much.
              </Text>
              <Text>
                <Text as="span" weight="semibold">Where it pays off.</Text>{" "}
                Per the patch docstring, site 2 saves ~13 ms / 3 steps of
                GPU time end-to-end, plus ~189 ms / 3 steps of RMSNorm GPU
                time (B0 vs B8 microbench). Most of that saving lives below
                the GEMM crit path here, but it gives back stream-0 SM
                budget that future elementwise/optimizer fusions can spend.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="info" title="Methodology">
        Trace produced by torch.profiler with{" "}
        <Text as="span">record_shapes=false, with_stack=false</Text>, profile
        steps 16-19 on rank 2 only (active 4 steps). Numbers come from
        ProfilerStep #17 parsed by{" "}
        <Text as="span">.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py</Text>.
        The opaque <Text as="span">ncclDevKernel_Generic_1</Text> kernels are
        re-categorized via overlapping <Text as="span">c10d::*</Text> CPU events
        (54 of 226 ms re-labelled to nccl_ag; the remaining 128 ms stays as
        nccl_generic - these are short DDP bucket sub-collectives that
        finish inside one CPU op span). Run config and logs are kept under{" "}
        <Text as="span">small_llm_moe_pretraining/primus/run-trace/20260425_b_rmsnorm_trace/</Text>.
      </Callout>
    </Stack>
  );
}
