// Options for `npm run comparison`; defaults reproduce the paper figure.

import {
  arg,
  argList,
  integerArg,
  nonNegativeIntegerArg,
  numberArg,
  positiveArg,
} from '../util/runtime.js';
import { AMP, CORNERS } from './images.js';

export const DEFAULT_SCENES = Object.freeze(['glyph:@', 'svg:demos/comparison/rosette.svg']);
const FITS = ['viewbox', 'ink'];
const KNOWN = new Set(['scene', 'size', 'samples', 'amp', 'offset', 'fit', 'zoom', 'inset', 'corner', 'out']);

/** Resolve parsed `--key value` options (runtime.parseArgs) into the comparison settings. */
export function comparisonOptions(options) {
  const unknown = Object.keys(options).find((key) => !KNOWN.has(key));
  if (unknown !== undefined) throw new Error(`unknown option --${unknown}`);
  // Repeat --scene rather than split on commas (',' is a valid glyph).
  const scenes = argList(options, 'scene', [...DEFAULT_SCENES], { comma: false }).map(String);
  if (scenes.some((scene) => scene === 'true')) throw new Error('--scene requires a value');
  const fit = String(arg(options, 'fit', 'viewbox')).toLowerCase();
  if (!FITS.includes(fit)) throw new Error(`--fit must be one of ${FITS.join(', ')}`);
  const corner = String(arg(options, 'corner', 'tr')).toLowerCase();
  if (!CORNERS.includes(corner)) throw new Error(`--corner must be one of ${CORNERS.join(', ')}`);
  const out = arg(options, 'out', null);
  if (out === true) throw new Error('--out requires a directory');
  return {
    scenes,
    size: integerArg(options, 'size', 512), // square render size in px
    samples: integerArg(options, 'samples', 64), // truth grid: samples^2 points per pixel
    amp: positiveArg(options, 'amp', AMP), // error-map gain
    offset: numberArg(options, 'offset', 0), // sub-pixel phase of every scene, px
    fit, // svg scenes: fit the viewBox or the ink bbox
    crop: {
      zoom: integerArg(options, 'zoom', 2),
      inset: nonNegativeIntegerArg(options, 'inset', 40),
      corner,
    },
    out: out === null ? null : String(out),
  };
}
