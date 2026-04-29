import { useState } from "react";
import type { TilePayload } from "../types";
import { tileImagePath, type TileOrientation } from "../tileAssets";

type TileProps = {
  tile: TilePayload;
  orientation?: TileOrientation;
  small?: boolean;
  highlight?: "predicted" | "actual" | "called";
};

export function Tile({ tile, orientation = "self", small = false, highlight }: TileProps) {
  const [failed, setFailed] = useState(false);
  const className = ["tile", `tile-${orientation}`, small ? "tile-small" : "", highlight ? `tile-${highlight}` : ""]
    .filter(Boolean)
    .join(" ");

  if (failed) {
    return (
      <span className={`${className} tile-fallback`} title={tile.name}>
        {tile.name}
      </span>
    );
  }

  return (
    <img
      className={className}
      src={tileImagePath(tile.code, orientation)}
      alt={tile.name}
      title={tile.name}
      onError={() => setFailed(true)}
    />
  );
}
