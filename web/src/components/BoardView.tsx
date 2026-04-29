import type { BoardPayload, DecisionPayload, EventPayload, PlayerPayload, TilePayload } from "../types";
import { Tile } from "./Tile";

type BoardViewProps = {
  board: BoardPayload;
  currentEvent?: EventPayload;
  decision?: DecisionPayload | null;
};

const ORIENTATIONS = {
  0: "self",
  1: "right",
  2: "top",
  3: "left",
} as const;

function chunkRiver(river: PlayerPayload["river"]) {
  const chunks = [];
  for (let index = 0; index < river.length; index += 6) {
    chunks.push(river.slice(index, index + 6));
  }
  return chunks;
}

/** ツモ直後は手元13枚とツモ牌を隙間で分けて並べる（見た目は実機に近い）。 */
function splitHandForDisplay(hand: TilePayload[], lastDraw: BoardPayload["last_draw_tile"]): {
  main: TilePayload[];
  tsumo: TilePayload | null;
} {
  if (!lastDraw || lastDraw.id === undefined || hand.length !== 14) {
    return { main: hand, tsumo: null };
  }
  const drawId = lastDraw.id;
  const index = hand.findIndex((t) => t.id === drawId);
  if (index < 0) {
    return { main: hand, tsumo: null };
  }
  const main = hand.filter((_, i) => i !== index);
  const tsumo = hand[index]!;
  return { main, tsumo };
}

function PlayerArea({ player }: { player: PlayerPayload }) {
  const orientation = ORIENTATIONS[player.relative as keyof typeof ORIENTATIONS] ?? "self";
  const riverChunks = chunkRiver(player.river);
  const concealedCount = player.relative === 0 ? 0 : Math.max(7, 13 - player.melds.length * 3);

  return (
    <section className={`player-area player-${player.relative}`}>
      {concealedCount > 0 && (
        <div className="concealed-hand" aria-label={`${player.label} 手牌`}>
          {Array.from({ length: concealedCount }, (_, index) => (
            <span className="tile-back" key={`${player.seat}-back-${index}`} />
          ))}
        </div>
      )}
      <div className="player-header">
        <span className="player-label">{player.label}</span>
        <strong>{player.name}</strong>
        <span>{player.score.toLocaleString()}</span>
        {player.is_dealer && <span className="badge">親</span>}
        {player.reach_status === 2 && <span className="badge badge-reach">立直</span>}
      </div>
      <div className="river">
        {riverChunks.map((chunk, chunkIndex) => (
          <div className="river-line" key={`${player.seat}-river-${chunkIndex}`}>
            {chunk.map((tile, tileIndex) => (
              <Tile
                key={`${player.seat}-${chunkIndex * 6 + tileIndex}-${tile.id}`}
                tile={tile}
                orientation={orientation}
                small
                highlight={tile.called ? "called" : undefined}
              />
            ))}
          </div>
        ))}
      </div>
      {player.melds.length > 0 && (
        <div className="melds">
          {player.melds.map((meld, meldIndex) => (
            <div className="meld" key={`${player.seat}-meld-${meldIndex}`}>
              <span>{meld.type}</span>
              {meld.tiles.map((tile, tileIndex) => (
                <Tile key={`${tile.id}-${tileIndex}`} tile={tile} orientation={orientation} small />
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function candidateProbability(decision: DecisionPayload | null | undefined, tile: TilePayload): number | null {
  const candidate = decision?.prediction.candidates.find((item) => item.tile.kind === tile.kind);
  return candidate ? candidate.probability : null;
}

function eventText(event?: EventPayload): string {
  if (!event) {
    return "";
  }
  const player = event.player >= 0 ? `P${event.player}` : "";
  const tile = event.tile?.name ?? (event.hidden ? "非公開" : "");
  return [event.type, player, tile].filter(Boolean).join(" ");
}

export function BoardView({ board, currentEvent, decision }: BoardViewProps) {
  const actor = board.players.find((player) => player.is_actor);
  const { main: handMain, tsumo: handTsumo } = splitHandForDisplay(board.hand, board.last_draw_tile);

  return (
    <div className="board-card">
      <div className="round-info">
        <div>
          <strong>
            {board.round.wind_label}{board.round.number}局
          </strong>
          <span>{board.round.honba}本場</span>
          <span>供託 {board.round.kyotaku}</span>
          <span>巡目 {board.round.junme.toFixed(1)}</span>
          <span>残り山 {board.round.wall_tile_count}</span>
        </div>
        <div className="dora">
          <span>ドラ表示</span>
          {board.dora_indicators.map((tile, index) => (
            <Tile key={`${tile.id}-${index}`} tile={tile} small />
          ))}
        </div>
      </div>

      <div className="table-layout">
        {board.players.map((player) => (
          <PlayerArea key={player.seat} player={player} />
        ))}
        <div className="table-center">
          <div>
            {board.round.wind_label}
            {board.round.number}
          </div>
          <div>{board.round.honba}本場</div>
          <div>視点 {actor?.name ?? `P${board.actor}`}</div>
          {currentEvent && <div>{eventText(currentEvent)}</div>}
        </div>
      </div>

      <div className="hand-area">
        <div className="hand-label">AI手牌</div>
        <div className="hand-stack">
          {board.hand.length === 14 && decision ? (
            <div className="hand-prob-line" aria-label="打牌候補確率">
              {handMain.map((tile, index) => {
                const probability = candidateProbability(decision, tile);
                return (
                  <div className="hand-prob-cell" key={`prob-${tile.id}-${index}`}>
                    {probability === null ? null : (
                      <>
                        <span className="hand-prob-label">{(probability * 100).toFixed(0)}</span>
                        <span className="hand-prob-bar" style={{ height: `${Math.max(probability * 34, 3)}px` }} />
                      </>
                    )}
                  </div>
                );
              })}
              {handTsumo ? (
                <>
                  <div className="hand-prob-gap" aria-hidden />
                  <div className="hand-prob-cell hand-prob-tsumo">
                    {candidateProbability(decision, handTsumo) !== null ? (
                      <>
                        <span className="hand-prob-label">
                          {((candidateProbability(decision, handTsumo) ?? 0) * 100).toFixed(0)}
                        </span>
                        <span
                          className="hand-prob-bar"
                          style={{ height: `${Math.max((candidateProbability(decision, handTsumo) ?? 0) * 34, 3)}px` }}
                        />
                      </>
                    ) : null}
                  </div>
                </>
              ) : null}
            </div>
          ) : null}
          <div className="hand-tiles">
          <div className="hand-tiles-main">
            {handMain.map((tile, index) => (
              <Tile key={`${tile.id}-${index}`} tile={tile} />
            ))}
          </div>
          {handTsumo ? (
            <>
              <div className="hand-tsumo-gap" aria-hidden />
              <div className="hand-tiles-tsumo">
                <Tile tile={handTsumo} />
              </div>
            </>
          ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
