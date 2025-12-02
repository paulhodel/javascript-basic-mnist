import { GPU } from 'gpu.js';

const gpu = new GPU();

// C = A (n x m) * B (m x p)
const multiplyMatrix = gpu.createKernel(function(A, B, m) {
    let sum = 0;
    for (let k = 0; k < m; k++) {
        sum += A[this.thread.y][k] * B[k][this.thread.x];
    }
    return sum;
}).setOutput([512, 512]);

// C = A (n x m) * B (m x p)
function multiplyMatrixCPU(A, B) {
  const n = A.length;        // rows in A
  const m = A[0].length;     // cols in A = rows in B
  const p = B[0].length;     // cols in B

  const C = new Array(n);
  for (let i = 0; i < n; i++) {
    C[i] = new Array(p);
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

// Example data: 512x512
const n = 512, m = 512, p = 512;
const A = [];
const B = [];

for (let i = 0; i < n; i++) {
    A[i] = [];
    for (let j = 0; j < m; j++) {
        A[i][j] = Math.random();
    }
}

for (let i = 0; i < m; i++) {
    B[i] = [];
    for (let j = 0; j < p; j++) {
        B[i][j] = Math.random();
    }
}

console.time();

for (let i = 0; i < 10; i++) {
    const C = multiplyMatrixCPU(A, B, m);
}

console.timeEnd();