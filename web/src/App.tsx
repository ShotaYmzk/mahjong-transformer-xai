import { useEffect, useMemo, useState } from "react";
import { BoardView } from "./components/BoardView";
import { DecisionPanel } from "./components/DecisionPanel";
import { RoundNavigator } from "./components/RoundNavigator";
import type { DecisionPayload, DecisionReport, FramePayload } from "./types";
import "./styles.css";

const REPORT_PATH = "/reports/sample.json";

function findDecisionForItem(
  item: DecisionPayload | FramePayload | null,
  decisions: DecisionPayload[],
): DecisionPayload | null {
  if (!item) {
    return null;
  }
  if ("prediction" in item) {
    return item;
  }

  const sameContext = decisions.filter(
    (decision) =>
      decision.metadata.source_file === item.metadata.source_file &&
      decision.metadata.round_index === item.metadata.round_index &&
      decision.metadata.player_id === item.metadata.player_id,
  );
  return (
    sameContext.find((decision) => decision.metadata.event_index >= item.metadata.event_index) ??
    sameContext[sameContext.length - 1] ??
    null
  );
}

export default function App() {
  const [report, setReport] = useState<DecisionReport | null>(null);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(REPORT_PATH)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${REPORT_PATH}: ${response.status}`);
        }
        return response.json() as Promise<DecisionReport>;
      })
      .then((data) => {
        setReport(data);
        setIndex(0);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, []);

  useEffect(() => {
    if (!report) {
      return undefined;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName;
      if (event.isComposing || tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setIndex((current) => Math.max(current - 1, 0));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        const length = report.frames?.length || report.decisions.length;
        setIndex((current) => Math.min(current + 1, length - 1));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [report]);

  const items: Array<DecisionPayload | FramePayload> = report?.decisions ?? [];
  const current = items[index] ?? null;
  const panelDecision = findDecisionForItem(current, report?.decisions ?? []);
  const accuracyText = useMemo(() => {
    if (!report || report.decisions.length === 0) {
      return "-";
    }
    const correct = report.decisions.filter((item) => {
      const predicted = item.prediction.predicted;
      const actual = item.prediction.actual;
      return predicted && actual && predicted.tile.kind === actual.tile.kind;
    }).length;
    return `${correct} / ${report.decisions.length}`;
  }, [report]);

  if (error) {
    return (
      <main className="app-shell">
        <div className="empty-state">レポートを読み込めません: {error}</div>
      </main>
    );
  }

  if (!report || !current) {
    return (
      <main className="app-shell">
        <div className="empty-state">レポートを読み込み中...</div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Mahjong Transformer V2</p>
          <h1>AI判断可視化ビューア</h1>
        </div>
        <div className="summary-strip">
          <div>
            <span>局面数</span>
            <strong>{report.decisions.length}</strong>
          </div>
          <div>
            <span>一致数</span>
            <strong>{accuracyText}</strong>
          </div>
          <div>
            <span>checkpoint</span>
            <strong>{report.source.checkpoint}</strong>
          </div>
        </div>
      </header>

      <div className="viewer-grid">
        <BoardView
          board={current.board}
          currentEvent={"current_event" in current ? current.current_event : undefined}
          decision={panelDecision}
        />
        <RoundNavigator items={items} decisions={report.decisions} index={index} onChange={setIndex} />
        <DecisionPanel decision={panelDecision} currentEvent={"current_event" in current ? current.current_event : undefined} />
      </div>
    </main>
  );
}
