export type TilePayload = {
  id?: number;
  kind: number;
  name: string;
  code: string;
  is_aka?: boolean;
  representative_id?: number;
  player?: number;
  tsumogiri?: boolean;
  called?: boolean;
};

export type MeldPayload = {
  player: number;
  type: string;
  tiles: TilePayload[];
  consumed: TilePayload[];
  called_tile: TilePayload | null;
  from_who: number;
};

export type PlayerPayload = {
  seat: number;
  relative: number;
  label: string;
  name: string;
  score: number;
  is_dealer: boolean;
  is_actor: boolean;
  reach_status: number;
  reach_junme: number;
  river: TilePayload[];
  melds: MeldPayload[];
};

export type BoardPayload = {
  actor: number;
  round: {
    wind: number;
    wind_label: string;
    number: number;
    honba: number;
    kyotaku: number;
    dealer: number;
    junme: number;
    wall_tile_count: number;
  };
  players: PlayerPayload[];
  hand: TilePayload[];
  dora_indicators: TilePayload[];
  valid_action_kinds: number[];
  last_draw_tile: TilePayload | null;
};

export type EventPayload = {
  index: number;
  type: string;
  player: number;
  tile: TilePayload | null;
  hidden: boolean;
  junme: number;
  data0: number;
  data1: number;
  is_padding: boolean;
};

export type CandidatePayload = {
  tile: TilePayload;
  logit: number;
  probability: number;
  is_actual: boolean;
  rank: number;
};

export type DecisionPayload = {
  metadata: {
    source_file: string;
    round_index: number;
    event_index: number;
    player_id: number;
    junme: number;
    decision_source: string;
    is_tsumogiri: boolean;
    label_tile_id: number;
  };
  board: BoardPayload;
  prediction: {
    top_k: number;
    predicted: CandidatePayload | null;
    actual: CandidatePayload | null;
    actual_rank: number | null;
    candidates: CandidatePayload[];
    top_candidates: CandidatePayload[];
    probability_sum: number;
  };
};

export type FramePayload = {
  metadata: {
    source_file: string;
    round_index: number;
    event_index: number;
    player_id: number;
    event_type: string;
    event_player: number;
  };
  current_event: EventPayload;
  board: BoardPayload;
};

export type DecisionReport = {
  schema: string;
  source: {
    input: string;
    checkpoint: string;
    offset_files: number;
    limit_files: number | null;
    include_call_discards: boolean;
  };
  summary: Record<string, unknown>;
  frame_summary?: Record<string, unknown>;
  frames?: FramePayload[];
  decisions: DecisionPayload[];
};
