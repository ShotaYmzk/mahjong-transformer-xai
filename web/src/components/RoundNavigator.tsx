import type { DecisionPayload, FramePayload } from "../types";

type RoundNavigatorProps = {
  items: Array<DecisionPayload | FramePayload>;
  decisions: DecisionPayload[];
  index: number;
  onChange: (nextIndex: number) => void;
};

function findPreviousIndex<T>(items: T[], startIndex: number, predicate: (item: T, index: number) => boolean): number {
  for (let itemIndex = Math.min(startIndex - 1, items.length - 1); itemIndex >= 0; itemIndex -= 1) {
    if (predicate(items[itemIndex]!, itemIndex)) {
      return itemIndex;
    }
  }
  return -1;
}

export function RoundNavigator({ items, decisions, index, onChange }: RoundNavigatorProps) {
  const current = items[index];
  const isReplay = Boolean(items[0] && "current_event" in items[0]);
  const previousMismatch = isReplay
    ? -1
    : findPreviousIndex(decisions, index, (decision) => {
        const predicted = decision.prediction.predicted;
        const actual = decision.prediction.actual;
        return Boolean(predicted && actual && predicted.tile.kind !== actual.tile.kind);
      });
  const nextMismatch = isReplay
    ? -1
    : decisions.findIndex((decision, decisionIndex) => {
        const predicted = decision.prediction.predicted;
        const actual = decision.prediction.actual;
        return decisionIndex > index && predicted && actual && predicted.tile.kind !== actual.tile.kind;
      });
  const previousLowConfidence = isReplay
    ? -1
    : findPreviousIndex(decisions, index, (decision) => (decision.prediction.predicted?.probability ?? 1) < 0.4);
  const nextLowConfidence = isReplay
    ? -1
    : decisions.findIndex(
        (decision, decisionIndex) => decisionIndex > index && (decision.prediction.predicted?.probability ?? 1) < 0.4,
      );

  return (
    <nav className="navigator">
      <button onClick={() => onChange(Math.max(index - 1, 0))} disabled={index === 0}>
        &lt; Prev
      </button>
      <button onClick={() => onChange(Math.min(index + 1, items.length - 1))} disabled={index >= items.length - 1}>
        Next &gt;
      </button>
      <button onClick={() => previousMismatch >= 0 && onChange(previousMismatch)} disabled={previousMismatch < 0}>
        &lt; Prev Error
      </button>
      <button onClick={() => nextMismatch >= 0 && onChange(nextMismatch)} disabled={nextMismatch < 0}>
        Next Error &gt;
      </button>
      <button onClick={() => previousLowConfidence >= 0 && onChange(previousLowConfidence)} disabled={previousLowConfidence < 0}>
        &lt; Prev Choice
      </button>
      <button onClick={() => nextLowConfidence >= 0 && onChange(nextLowConfidence)} disabled={nextLowConfidence < 0}>
        Next Choice &gt;
      </button>
      <div className="position">
        <strong>
          {index + 1} / {items.length}
        </strong>
        <span>
          {current.board.round.wind_label}
          {current.board.round.number}局 {current.board.round.junme.toFixed(1)}巡目
          {"current_event" in current ? ` ${current.current_event.type}` : ""}
        </span>
      </div>
      <button disabled>Options</button>
    </nav>
  );
}
