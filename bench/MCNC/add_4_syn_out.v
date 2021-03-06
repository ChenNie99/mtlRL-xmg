module addition(q, a, b);
  wire _00_;
  wire _01_;
  wire _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire _07_;
  wire _temp0_;
  wire _temp1_;
  wire _temp2_;
  wire _temp3_;
  wire _temp4_;
  wire _temp5_;
  wire _temp6_;
  wire _temp7_;
  input [3:0] a;
  input [3:0] b;
  output [3:0] q;
  assign _temp0_ = b[1] ^ a[1];
  assign _00_ = ~_temp0_;
  assign _temp1_ = b[0] & a[0];
  assign _01_ = ~_temp1_;
  assign q[1] = _01_ ^ _00_;
  assign _02_ = b[2] ^ a[2];
  assign _temp2_ = b[1] & a[1];
  assign _03_ = ~_temp2_;
  assign _temp3_ = _01_ | _00_;
  assign _temp4_ = _temp3_ & _03_;
  assign _04_ = ~_temp4_;
  assign q[2] = _04_ ^ _02_;
  assign _temp5_ = b[3] ^ a[3];
  assign _05_ = ~_temp5_;
  assign _06_ = b[2] & a[2];
  assign _temp6_ = _04_ & _02_;
  assign _temp7_ = _temp6_ | _06_;
  assign _07_ = ~_temp7_;
  assign q[3] = _07_ ^ _05_;
  assign q[0] = b[0] ^ a[0];
endmodule
