import type { DecisionPayload, EventPayload } from "../types";
import { Tile } from "./Tile";

type DecisionPanelProps = {
  decision: DecisionPayload | null;
  currentEvent?: EventPayload;
};

function formatPercent(value: number): string {
  return (value * 100).toFixed(2);
}

function eventText(event?: EventPayload): string {
  if (!event) {
    return "予測対象の局面を待機中";
  }
  const player = event.player >= 0 ? `P${event.player}` : "";
  const tile = event.tile?.name ?? (event.hidden ? "非公開" : "");
  return [event.type, player, tile].filter(Boolean).join(" ");
}

export function DecisionPanel({ decision, currentEvent }: DecisionPanelProps) {
  if (!decision) {
    return (
      <aside className="decision-panel prediction-panel">
        <div className="prediction-empty">
          <span className="eyebrow">Prediction</span>
          <strong>{eventText(currentEvent)}</strong>
          <p>このフレームに対応するAI打牌予測がありません。</p>
        </div>
      </aside>
    );
  }

  const { prediction, metadata } = decision;
  const predicted = prediction.predicted;
  const actual = prediction.actual;
  const isCorrect = Boolean(predicted && actual && predicted.tile.kind === actual.tile.kind);
  const rows = prediction.candidates.length ? prediction.candidates : prediction.top_candidates;

  return (
    <aside className="decision-panel prediction-panel">
      <div className="prediction-summary">
        <div className="prediction-side">
          <span>Player</span>
          <strong>Cut</strong>
          {actual ? <Tile tile={actual.tile} small highlight="actual" /> : <span className="empty-tile">-</span>}
        </div>
        <div className="prediction-side">
          <span>AI</span>
          <strong>Cut</strong>
          {predicted ? <Tile tile={predicted.tile} small highlight="predicted" /> : <span className="empty-tile">-</span>}
        </div>
      </div>

      <div className="prediction-meta">
        <span className={isCorrect ? "status status-ok" : "status status-miss"}>{isCorrect ? "一致" : "不一致"}</span>
        <span>{metadata.is_tsumogiri ? "ツモ切り" : "手出し"}</span>
        <span>
          R{metadata.round_index} / E{metadata.event_index}
        </span>
      </div>

      <div className="action-list">
        <div className="action-head">
          <span>Action</span>
          <span>Q</span>
          <span>P</span>
        </div>
        {rows.map((candidate) => (
          <div
            className={[
              "action-row",
              candidate.rank === 1 ? "action-top" : "",
              candidate.is_actual ? "action-actual" : "",
              predicted && predicted.tile.kind === candidate.tile.kind ? "action-predicted" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            key={candidate.tile.kind}
          >
            <div className="action-cell">
              <span>Cut</span>
              <Tile tile={candidate.tile} small highlight={candidate.is_actual ? "actual" : undefined} />
            </div>
            <code>{candidate.logit.toFixed(2)}</code>
            <strong>{formatPercent(candidate.probability)}</strong>
          </div>
        ))}
      </div>
    </aside>
  );
}
