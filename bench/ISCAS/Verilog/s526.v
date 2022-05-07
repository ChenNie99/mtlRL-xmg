module s526(VDD,CK,G0,G1,G147,G148,G198,G199,G2,G213,G214);
input VDD,CK,G0,G1,G2;
output G198,G213,G147,G199,G214,G148;

  wire G10,G60,G11,G61,G12,G62,G13,G69,G14,G79,G15,G84,G16,G89,G17,G96,G18,
    G101,G19,G106,G20,G115,G21,G127,G22,G137,G23,G167,G24,G173,G25,G179,G26,
    G183,G27,G188,G28,G194,G29,G200,G30,G206,G59,G211,G65,G72,G83,G85,G90,G94,
    G104,G107,G112,G116,G122,G124,G126,G131,G135,G136,G140,G141,G145,II285,
    II288,G157,G163,G168,G172,G174,G177,G180,G184,G189,G193,G195,II340,II343,
    G201,G205,G202,II351,G207,G212,G208,II359,II362,II365,II368,G34,G123,G35,
    G36,G38,G39,G40,G44,G45,G46,G47,G51,G52,G54,G55,G56,G57,G58,G76,G77,G78,
    G80,G81,G82,G87,G86,G92,G91,G93,G98,G108,G97,G99,G102,G118,G103,G109,G113,
    G110,G111,G119,G117,G120,G121,G125,G138,G142,G139,G146,G143,G144,G155,G156,
    G169,G170,G171,G175,G176,G181,G186,G185,G191,G190,G192,G196,G203,G204,G209,
    G210,G32,G31,G33,G42,G43,G48,G49,G50,G53,G67,G63,G64,G71,G68,G66,G74,G70,
    G75,G73,G133,G128,G129,G130,G134,G132,G149,G150,G151,G152,G153,G154,G158,
    G159,G160,G161,G162,G164,G165,G166,G41,G37,G88,G100,G178,G182,G187,G197,
    G95,G105,G114;

  FD1 DFF_0(CK,G10,G60);
  FD1 DFF_1(CK,G11,G61);
  FD1 DFF_2(CK,G12,G62);
  FD1 DFF_3(CK,G13,G69);
  FD1 DFF_4(CK,G14,G79);
  FD1 DFF_5(CK,G15,G84);
  FD1 DFF_6(CK,G16,G89);
  FD1 DFF_7(CK,G17,G96);
  FD1 DFF_8(CK,G18,G101);
  FD1 DFF_9(CK,G19,G106);
  FD1 DFF_10(CK,G20,G115);
  FD1 DFF_11(CK,G21,G127);
  FD1 DFF_12(CK,G22,G137);
  FD1 DFF_13(CK,G23,G167);
  FD1 DFF_14(CK,G24,G173);
  FD1 DFF_15(CK,G25,G179);
  FD1 DFF_16(CK,G26,G183);
  FD1 DFF_17(CK,G27,G188);
  FD1 DFF_18(CK,G28,G194);
  FD1 DFF_19(CK,G29,G200);
  FD1 DFF_20(CK,G30,G206);
  IV  NOT_0(G59,G211);
  IV  NOT_1(G65,G12);
  IV  NOT_2(G72,G13);
  IV  NOT_3(G83,G10);
  IV  NOT_4(G85,G15);
  IV  NOT_5(G90,G14);
  IV  NOT_6(G94,G16);
  IV  NOT_7(G104,G18);
  IV  NOT_8(G107,G11);
  IV  NOT_9(G112,G19);
  IV  NOT_10(G116,G17);
  IV  NOT_11(G122,G30);
  IV  NOT_12(G124,G20);
  IV  NOT_13(G126,G59);
  IV  NOT_14(G131,G21);
  IV  NOT_15(G135,G20);
  IV  NOT_16(G136,G12);
  IV  NOT_17(G140,G21);
  IV  NOT_18(G141,G29);
  IV  NOT_19(G145,G22);
  IV  NOT_20(II285,G23);
  IV  NOT_21(G147,II285);
  IV  NOT_22(II288,G24);
  IV  NOT_23(G148,II288);
  IV  NOT_24(G157,G18);
  IV  NOT_25(G163,G20);
  IV  NOT_26(G168,G23);
  IV  NOT_27(G172,G21);
  IV  NOT_28(G174,G24);
  IV  NOT_29(G177,G13);
  IV  NOT_30(G180,G25);
  IV  NOT_31(G184,G12);
  IV  NOT_32(G189,G193);
  IV  NOT_33(G195,G28);
  IV  NOT_34(II340,G25);
  IV  NOT_35(G198,II340);
  IV  NOT_36(II343,G26);
  IV  NOT_37(G199,II343);
  IV  NOT_38(G201,G205);
  IV  NOT_39(G202,G29);
  IV  NOT_40(II351,G2);
  IV  NOT_41(G205,II351);
  IV  NOT_42(G207,G212);
  IV  NOT_43(G208,G30);
  IV  NOT_44(II359,G0);
  IV  NOT_45(G211,II359);
  IV  NOT_46(II362,G1);
  IV  NOT_47(G212,II362);
  IV  NOT_48(II365,G27);
  IV  NOT_49(G213,II365);
  IV  NOT_50(II368,G28);
  IV  NOT_51(G214,II368);
  AN2 AND2_0(G34,G122,G123);
  AN4 AND4_0(G35,G10,G107,G90,G15);
  AN2 AND2_1(G36,G122,G123);
  AN2 AND2_2(G38,G122,G123);
  AN2 AND2_3(G39,G65,G21);
  AN2 AND2_4(G40,G12,G131);
  AN2 AND2_5(G44,G59,G94);
  AN2 AND2_6(G45,G122,G59);
  AN2 AND2_7(G46,G116,G112);
  AN2 AND2_8(G47,G116,G18);
  AN3 AND3_0(G51,G59,G16,G17);
  AN2 AND2_9(G52,G59,G18);
  AN2 AND2_10(G54,G17,G104);
  AN2 AND2_11(G55,G116,G18);
  AN2 AND2_12(G56,G17,G112);
  AN4 AND4_1(G57,G59,G16,G17,G18);
  AN2 AND2_13(G58,G59,G19);
  AN3 AND3_1(G76,G10,G90,G15);
  AN2 AND2_14(G77,G10,G11);
  AN2 AND2_15(G78,G83,G107);
  AN3 AND3_2(G80,G10,G11,G14);
  AN2 AND2_16(G81,G83,G90);
  AN2 AND2_17(G82,G107,G90);
  AN2 AND2_18(G87,G85,G86);
  AN4 AND4_2(G92,G90,G107,G10,G91);
  AN3 AND3_3(G93,G94,G122,G123);
  AN4 AND4_3(G98,G107,G10,G108,G97);
  AN3 AND3_4(G99,G116,G122,G123);
  AN4 AND4_4(G102,G18,G17,G16,G118);
  AN3 AND3_5(G103,G104,G122,G123);
  AN4 AND4_5(G109,G107,G10,G108,G113);
  AN3 AND3_6(G110,G112,G122,G123);
  AN3 AND3_7(G111,G16,G30,G113);
  AN4 AND4_6(G119,G116,G16,G117,G118);
  AN3 AND3_8(G120,G124,G122,G123);
  AN2 AND2_19(G121,G124,G125);
  AN4 AND4_7(G138,G140,G20,G141,G142);
  AN2 AND2_20(G139,G145,G146);
  AN4 AND4_8(G143,G140,G20,G141,G142);
  AN2 AND2_21(G144,G145,G146);
  AN3 AND3_9(G155,G21,G13,G26);
  AN3 AND3_10(G156,G163,G172,G13);
  AN2 AND2_22(G169,G13,G168);
  AN2 AND2_23(G170,G184,G177);
  AN2 AND2_24(G171,G172,G184);
  AN2 AND2_25(G175,G174,G12);
  AN2 AND2_26(G176,G177,G12);
  AN3 AND3_11(G181,G180,G13,G21);
  AN3 AND3_12(G186,G184,G189,G185);
  AN2 AND2_27(G191,G189,G190);
  AN2 AND2_28(G192,G18,G193);
  AN2 AND2_29(G196,G195,G13);
  AN2 AND2_30(G203,G201,G202);
  AN2 AND2_31(G204,G205,G29);
  AN2 AND2_32(G209,G207,G208);
  AN2 AND2_33(G210,G212,G30);
  OR2 OR2_0(G32,G30,G31);
  OR2 OR2_1(G33,G72,G12);
  OR4 OR4_0(G42,G83,G107,G90,G85);
  OR3 OR3_0(G43,G83,G11,G14);
  OR4 OR4_1(G48,G122,G94,G18,G112);
  OR3 OR3_1(G49,G122,G94,G116);
  OR2 OR2_2(G50,G16,G17);
  OR4 OR4_2(G53,G83,G11,G14,G85);
  OR4 OR4_3(G67,G211,G63,G64,G71);
  OR3 OR3_2(G68,G65,G211,G66);
  OR4 OR4_4(G74,G211,G125,G70,G71);
  OR3 OR3_3(G75,G72,G211,G73);
  OR4 OR4_5(G133,G211,G128,G129,G130);
  OR3 OR3_4(G134,G131,G211,G132);
  OR4 OR4_6(G149,G20,G21,G12,G177);
  OR2 OR2_3(G150,G184,G25);
  OR2 OR2_4(G151,G184,G13);
  OR3 OR3_5(G152,G163,G21,G12);
  OR2 OR2_5(G153,G172,G27);
  OR2 OR2_6(G154,G184,G27);
  OR4 OR4_7(G158,G193,G184,G177,G26);
  OR2 OR2_7(G159,G189,G157);
  OR2 OR2_8(G160,G184,G13);
  OR2 OR2_9(G161,G20,G13);
  OR2 OR2_10(G162,G21,G12);
  OR4 OR4_8(G164,G20,G21,G12,G177);
  OR3 OR3_6(G165,G163,G172,G13);
  OR3 OR3_7(G166,G172,G177,G24);
  ND4 NAND4_0(G41,G104,G116,G16,G37);
  ND2 NAND2_0(G62,G67,G68);
  ND3 NAND3_0(G63,G104,G116,G16);
  ND4 NAND4_1(G64,G65,G21,G20,G19);
  ND2 NAND2_1(G69,G74,G75);
  ND4 NAND4_2(G70,G72,G12,G21,G20);
  ND3 NAND3_1(G86,G14,G11,G10);
  ND3 NAND3_2(G88,G42,G43,G59);
  ND4 NAND4_3(G100,G48,G49,G50,G59);
  ND2 NAND2_2(G118,G53,G122);
  ND4 NAND4_4(G123,G15,G90,G107,G10);
  ND4 NAND4_5(G125,G19,G104,G116,G16);
  ND2 NAND2_3(G127,G133,G134);
  ND2 NAND2_4(G128,G116,G16);
  ND4 NAND4_6(G129,G131,G20,G19,G104);
  ND2 NAND2_5(G130,G32,G33);
  ND4 NAND4_7(G146,G140,G135,G29,G142);
  ND4 NAND4_8(G178,G164,G165,G166,G189);
  ND4 NAND4_9(G182,G149,G150,G151,G189);
  ND2 NAND2_6(G187,G158,G159);
  ND4 NAND4_10(G190,G152,G153,G154,G13);
  ND4 NAND4_11(G197,G160,G161,G162,G189);
  NR4 NOR4_0(G31,G85,G14,G11,G83);
  NR2 NOR2_0(G37,G124,G112);
  NR2 NOR2_1(G60,G10,G211);
  NR4 NOR4_1(G61,G76,G77,G78,G211);
  NR4 NOR4_2(G66,G36,G131,G124,G125);
  NR2 NOR2_2(G71,G35,G30);
  NR4 NOR4_3(G73,G38,G39,G40,G41);
  NR4 NOR4_4(G79,G80,G81,G82,G211);
  NR2 NOR2_3(G84,G87,G88);
  NR3 NOR3_0(G89,G92,G93,G95);
  NR2 NOR2_4(G91,G94,G85);
  NR2 NOR2_5(G95,G44,G45);
  NR3 NOR3_1(G96,G98,G99,G100);
  NR2 NOR2_6(G97,G46,G47);
  NR3 NOR3_2(G101,G102,G103,G105);
  NR2 NOR2_7(G105,G51,G52);
  NR4 NOR4_5(G106,G109,G110,G111,G114);
  NR3 NOR3_3(G108,G94,G85,G14);
  NR3 NOR3_4(G113,G54,G55,G56);
  NR2 NOR2_8(G114,G57,G58);
  NR4 NOR4_6(G115,G119,G120,G121,G126);
  NR3 NOR3_5(G117,G124,G112,G18);
  NR3 NOR3_6(G132,G34,G124,G125);
  NR3 NOR3_7(G137,G138,G139,G211);
  NR2 NOR2_9(G142,G13,G136);
  NR4 NOR4_7(G167,G169,G170,G171,G193);
  NR3 NOR3_8(G173,G175,G176,G178);
  NR2 NOR2_10(G179,G181,G182);
  NR2 NOR2_11(G183,G186,G187);
  NR2 NOR2_12(G185,G155,G156);
  NR2 NOR2_13(G188,G191,G192);
  NR2 NOR2_14(G193,G143,G144);
  NR2 NOR2_15(G194,G196,G197);
  NR3 NOR3_9(G200,G203,G204,G211);
  NR3 NOR3_10(G206,G209,G210,G211);

endmodule
