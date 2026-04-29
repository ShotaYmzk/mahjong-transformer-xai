const HONOR_ASSET_CODES: Record<string, string> = {
  east: "1z",
  south: "2z",
  west: "3z",
  north: "4z",
  white: "5z",
  green: "6z",
  red: "7z",
};

const ORIENTATION_DIRS = {
  self: "self_bottom",
  right: "shimocha",
  top: "toimen",
  left: "kamicha",
} as const;

export type TileOrientation = keyof typeof ORIENTATION_DIRS;

export function tileImagePath(code: string, orientation: TileOrientation = "self"): string {
  const assetCode = HONOR_ASSET_CODES[code] ?? code;
  return `/tiles/${ORIENTATION_DIRS[orientation]}/${assetCode}.png`;
}
