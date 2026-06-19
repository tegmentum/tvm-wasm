/** Port of `tvm-core::PlacementPolicy` (crates/tvm-core/src/policy.rs). */
import { RegionKind, Residency } from "./types.js";

export interface PlacementPolicy {
  initialResidency: Residency;
  pinnable: boolean;
  spillable: boolean;
}

export function policyForKind(kind: RegionKind): PlacementPolicy {
  switch (kind) {
    case RegionKind.HotHeap:
    case RegionKind.CodeCache:
    case RegionKind.DeviceState:
      return { initialResidency: Residency.Hot, pinnable: true, spillable: false };
    case RegionKind.ObjectArena:
    case RegionKind.BlobArena:
      return { initialResidency: Residency.Hot, pinnable: false, spillable: true };
    case RegionKind.PageStore:
      return { initialResidency: Residency.Warm, pinnable: false, spillable: true };
    case RegionKind.Scratch:
      return { initialResidency: Residency.Hot, pinnable: false, spillable: false };
  }
}
