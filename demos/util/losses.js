import { loadTarget, targetFromSource } from './image.js';
import { LossClient } from './loss-client.js';
import { tonemapImage } from '../../js/tonemap.js';
import { arg, nonNegativeIntegerArg } from './runtime.js';

const CLIP_MODEL = 'ViT-B-32-quickgelu';
// CLIPAG (perceptually aligned gradients) is an adversarially finetuned
// ViT-B-32 without quickgelu; the server resolves the "clipag" weights tag.
const CLIPAG_MODEL = 'ViT-B-32';
const clipModel = (config) => (config.clipag ? CLIPAG_MODEL : CLIP_MODEL);

// The perspective augmentation pads with "outside the canvas", which in
// display space is the background -- not literal white, which a tonemapped
// render can never reach (so every crop would contain values brighter than
// anything the scene can produce). Mapped with the seed exposure/white; the
// learned scalars drift during the fit, but the fill is an augmentation
// detail, not a loss target.
function displayFill(config, background) {
  const bg = background ?? [1, 1, 1];
  if (!config.tonemap || config.tonemap === 'none') return [bg[0], bg[1], bg[2]];
  const px = tonemapImage(
    config.tonemap,
    Float32Array.of(bg[0], bg[1], bg[2], 1),
    config.exposure,
    config.white,
  );
  return [px[0], px[1], px[2]];
}

// A loss descriptor owns everything that varies with the objective: how the
// subject is named, whether an image target is loaded, how the per-step loss is
// produced, and the loss-specific fields written to config.json. The shared
// runner dispatches to these instead of branching on the loss kind.
export const LOSSES = {
  l2: {
    subjectKey: 'target',
    // No imagery is tracked: the default target is the Färlev fixture, which a
    // fresh clone downloads once (fixtures/manifests/farlev-highres.json).
    subjectDefault: 'fixtures/wikimedia/farlev-dip-in-road.jpg',
    subjectDefaultHint: 'run `npm run fixtures:farlev` to download it ' +
      '(14 MB, SHA256-verified), or pass --target=PATH',
    splitSubjects: true,
    loadsTarget: true,
    parse() {
      return {};
    },
    async loadTarget({ config, width, height }) {
      return config.targetSource
        ? targetFromSource(config.targetSource, width, height)
        : loadTarget(config.subject, width, height);
    },
    async setup() {
      return null;
    },
    outputFields(config) {
      return { target: config.subject };
    },
  },
  clip: {
    subjectKey: 'prompt',
    subjectDefault: 'a hot air balloon festival',
    splitSubjects: false,
    loadsTarget: false,
    parse(options) {
      return {
        augs: nonNegativeIntegerArg(options, 'augs', 4),
        lossUrl: String(arg(options, 'loss-url', 'ws://127.0.0.1:8765')),
        clipag: 'clipag' in options,
        clipWeights: String(
          arg(options, 'clip-weights', 'clipag' in options ? 'clipag' : 'openai'),
        ),
      };
    },
    async loadTarget() {
      return null;
    },
    async setup({ config, width, height, background }) {
      const client = new LossClient(config.lossUrl);
      const ready = await client.connect({
        loss: 'clip',
        prompt: config.subject,
        augs: config.augs,
        seed: config.seed,
        model: clipModel(config),
        pretrained: config.clipWeights,
        fill: displayFill(config, background),
      });
      if (!config.quiet) console.log(`CLIP server: ${ready.device}`);
      return {
        client,
        loss: (image, step) => client.grad(image, width, height, step),
      };
    },
    outputFields(config) {
      return {
        prompt: config.subject,
        augs: config.augs,
        lossUrl: config.lossUrl,
        clipModel: clipModel(config),
        clipWeights: config.clipWeights,
      };
    },
  },
};
