import { createSceneRenderer, renderSettings } from './render.js';

// Present Windfoil's float image buffer directly to a WebGPU canvas.
export class Preview {
  constructor(device, canvas) {
    this.device = device;
    this.canvas = canvas;
    this.renderer = null;
    this.context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: 'opaque' });
    this.dimensions = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read> image: array<vec4f>;
      @group(0) @binding(1) var<uniform> dimensions: vec4u;
      @vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let points = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(points[index], 0, 1);
      }
      @fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
        let xy = min(vec2u(position.xy), dimensions.xy - vec2u(1));
        return vec4f(clamp(image[xy.y * dimensions.x + xy.x].rgb, vec3f(0), vec3f(1)), 1);
      }
    ` });
    this.pipeline = device.createRenderPipeline({
      layout: 'auto', vertex: { module, entryPoint: 'vertex' },
      fragment: { module, entryPoint: 'fragment', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  async render(snapshot, width, height, isCurrent = () => true) {
    const previous = this.renderer;
    const rebuild = !previous || previous.width !== width || previous.height !== height ||
      previous.blend !== snapshot.blend || snapshot.maxShapes > previous.maxShapes ||
      snapshot.maxPieces > previous.maxPieces || snapshot.maxCurves > previous.maxCurves;
    const next = rebuild ? await createSceneRenderer(this.device, snapshot, width, height) : previous;
    if (!isCurrent()) {
      if (rebuild) next.destroy();
      return false;
    }
    this.renderer = next;
    if (rebuild) previous?.destroy();
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    next.uploadScene(snapshot.scene, renderSettings(snapshot, width, height));
    next.forwardNoRead();
    this.draw(next);
    return true;
  }

  clear() {
    this.draw(null);
  }

  draw(renderer) {
    if (renderer && this.image !== renderer.imageBuf) {
      this.image = renderer.imageBuf;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.image } },
          { binding: 1, resource: { buffer: this.dimensions } },
        ],
      });
      this.device.queue.writeBuffer(this.dimensions, 0,
        new Uint32Array([renderer.width, renderer.height, 0, 0]));
    }
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.context.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 1, b: 1, a: 1 },
    }] });
    if (renderer) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
