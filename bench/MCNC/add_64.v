/* addition */
parameter N = 64;
module addition(q, a, b);
output[N - 1:0] q;
input[N - 1:0] a, b;

assign q = a + b;

endmodule
