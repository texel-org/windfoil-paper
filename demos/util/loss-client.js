export class LossClient {
  constructor(url = 'ws://127.0.0.1:8765') {
    this.url = url;
    this.id = 0;
    this.queue = [];
    this.waiters = [];
  }

  async connect(config) {
    this.error = null;
    this.queue.length = 0;
    this.waiters.length = 0;
    this.socket = new WebSocket(this.url);
    this.socket.binaryType = 'arraybuffer';
    this.socket.onmessage = ({ data }) => {
      if (this.error) return;
      const waiter = this.waiters.shift();
      waiter ? waiter.resolve(data) : this.queue.push(data);
    };
    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      const fail = (error) => {
        this.fail(error);
        reject(error);
      };
      this.socket.onerror = () => fail(new Error(`CLIP loss server connection failed: ${this.url}`));
      this.socket.onclose = ({ code, reason }) =>
        fail(
          new Error(
            `CLIP loss server closed${code ? ` (${code}${reason ? `: ${reason}` : ''})` : ''}`,
          ),
        );
    });
    this.socket.send(JSON.stringify({ type: 'config', ...config }));
    const reply = JSON.parse(await this.receive());
    if (reply.type === 'error') throw new Error(reply.message);
    if (reply.type !== 'ready') throw new Error(`unexpected loss-server reply: ${reply.type}`);
    return reply;
  }

  async grad(image, width, height, step) {
    const id = ++this.id;
    this.socket.send(JSON.stringify({ type: 'grad', id, step, w: width, h: height }));
    this.socket.send(image.buffer.slice(image.byteOffset, image.byteOffset + image.byteLength));
    const header = JSON.parse(await this.receive());
    const data = await this.receive();
    if (header.type === 'error' || header.loss == null) {
      throw new Error(header.message ?? 'CLIP loss became non-finite');
    }
    return { loss: header.loss, dLdI: new Float32Array(data) };
  }

  receive() {
    if (this.queue.length) return Promise.resolve(this.queue.shift());
    if (this.error) return Promise.reject(this.error);
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  fail(error) {
    this.error ??= error;
    this.queue.length = 0;
    for (const waiter of this.waiters.splice(0)) waiter.reject(this.error);
  }

  close() {
    this.fail(new Error('CLIP loss client closed'));
    this.socket?.close();
    this.socket = null;
  }
}
