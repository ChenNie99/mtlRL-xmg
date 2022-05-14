module s820(VDD,CK,G0,G1,G10,G11,G12,G13,G14,G15,G16,G18,G2,G288,G290,G292,
  G296,G298,
  G3,G300,G302,G310,G312,G315,G322,G325,G327,G4,G43,G45,G47,G49,G5,G53,G55,G6,
  G7,G8,G9);
input VDD,CK,G0,G1,G2,G3,G4,G5,G6,G7,G8,G9,G10,G11,G12,G13,G14,G15,G16,G18;
output G290,G327,G47,G55,G288,G296,G310,G312,G325,G300,G43,G53,G298,G315,G322,
  G49,G45,G292,G302;

  wire G38,G90,G39,G93,G40,G96,G41,G99,G42,G102,G245,G323,G181,G256,G130,G203,
    G202,G112,G198,G171,G172,G168,G201,G267,G317,G281,G313,G328,G88,G91,G94,
    G97,G100,G280,G318,II127,G228,II130,G229,II133,G231,II198,G247,G143,G161,
    G162,G163,G188,G189,G190,G195,G215,G120,G250,G118,G166,G199,G170,G169,G129,
    G265,G142,G279,G103,G164,G167,G191,G200,G214,G234,G283,G141,G140,G127,G160,
    G187,G193,G194,G213,G235,G249,G268,G276,G282,G117,G277,G278,G121,G128,G232,
    G233,G251,G252,G271,G270,G210,G209,G226,G225,G175,G176,G197,G196,G263,G262,
    G150,G147,G148,G149,G158,G157,G185,G184,G174,G173,G211,G212,G223,G222,G272,
    G274,G264,G266,G294,G293,G152,G154,G218,G216,G217,G151,G153,G273,G275,G258,
    G257,G219,G220,G259,G260,G89,G92,G95,G98,G101,G126,G124,G125,G107,G145,
    G243,G111,G144,G239,G287,G115,G183,G237,G246,G113,G132,G133,G182,G238,G241,
    G136,G116,G286,G108,G109,G240,G242,G244,G110,G134,G135,G114,G236,G248,G321,
    G319,G180,G178,G78,G73,G74,G285,G284,G63,G59,G106,G105,G308,G304,G320,G316,
    G52,G50,G139,G137,G255,G253,G207,G204,G205,G309,G305,G62,G57,G58,G307,G303,
    G85,G81,G67,G177,G70,G65,G66,G155,G79,G75,G64,G60,G72,G68,G71,G86,G82,G80,
    G76,G87,G83,G123,G295,G291,G329,G48,G56,G289,G297,G311,G314,G326,G301,G119,
    G44,G54,G156,G299,G179,G224,G227,G131,G269,G46,G122,G69,G306,G138,G84,G254,
    G51,G61,G146,G206,G77,G165,G192,G104,G324,G159,G186,G221,G261;

  FD1 DFF_0(CK,G38,G90);
  FD1 DFF_1(CK,G39,G93);
  FD1 DFF_2(CK,G40,G96);
  FD1 DFF_3(CK,G41,G99);
  FD1 DFF_4(CK,G42,G102);
  IV  NOT_0(G245,G0);
  IV  NOT_1(G323,G1);
  IV  NOT_2(G181,G2);
  IV  NOT_3(G256,G4);
  IV  NOT_4(G130,G5);
  IV  NOT_5(G203,G6);
  IV  NOT_6(G202,G7);
  IV  NOT_7(G112,G8);
  IV  NOT_8(G198,G9);
  IV  NOT_9(G171,G10);
  IV  NOT_10(G172,G11);
  IV  NOT_11(G168,G12);
  IV  NOT_12(G201,G13);
  IV  NOT_13(G267,G15);
  IV  NOT_14(G317,G40);
  IV  NOT_15(G281,G16);
  IV  NOT_16(G313,G41);
  IV  NOT_17(G328,G42);
  IV  NOT_18(G88,G18);
  IV  NOT_19(G91,G18);
  IV  NOT_20(G94,G18);
  IV  NOT_21(G97,G18);
  IV  NOT_22(G100,G18);
  IV  NOT_23(G280,G38);
  IV  NOT_24(G318,G39);
  IV  NOT_25(II127,G38);
  IV  NOT_26(G228,II127);
  IV  NOT_27(II130,G15);
  IV  NOT_28(G229,II130);
  IV  NOT_29(II133,G313);
  IV  NOT_30(G231,II133);
  IV  NOT_31(II198,G38);
  IV  NOT_32(G247,II198);
  AN2 AND2_0(G143,G40,G4);
  AN2 AND2_1(G161,G3,G42);
  AN2 AND2_2(G162,G1,G42);
  AN2 AND2_3(G163,G41,G42);
  AN2 AND2_4(G188,G3,G42);
  AN2 AND2_5(G189,G1,G42);
  AN2 AND2_6(G190,G41,G42);
  AN2 AND2_7(G195,G41,G42);
  AN2 AND2_8(G215,G41,G42);
  AN3 AND3_0(G120,G39,G40,G42);
  AN3 AND3_1(G250,G39,G40,G42);
  AN3 AND3_2(G118,G245,G38,G39);
  AN3 AND3_3(G166,G245,G38,G42);
  AN3 AND3_4(G199,G245,G38,G42);
  AN2 AND2_9(G170,G171,G172);
  AN2 AND2_10(G169,G172,G168);
  AN2 AND2_11(G129,G39,G317);
  AN2 AND2_12(G265,G317,G267);
  AN2 AND2_13(G142,G40,G281);
  AN2 AND2_14(G279,G281,G42);
  AN2 AND2_15(G103,G313,G38);
  AN2 AND2_16(G164,G42,G313);
  AN3 AND3_5(G167,G256,G38,G313);
  AN2 AND2_17(G191,G42,G313);
  AN3 AND3_6(G200,G256,G38,G313);
  AN2 AND2_18(G214,G267,G16);
  AN4 AND4_0(G234,G15,G40,G313,G42);
  AN2 AND2_19(G283,G317,G313);
  AN4 AND4_1(G141,G317,G16,G323,G140);
  AN4 AND4_2(G127,G38,G39,G313,G328);
  AN3 AND3_7(G160,G5,G313,G328);
  AN3 AND3_8(G187,G5,G313,G328);
  AN2 AND2_20(G193,G11,G328);
  AN2 AND2_21(G194,G10,G328);
  AN3 AND3_9(G213,G16,G313,G328);
  AN2 AND2_22(G235,G317,G328);
  AN3 AND3_10(G249,G40,G41,G328);
  AN2 AND2_23(G268,G328,G267);
  AN3 AND3_11(G276,G0,G38,G328);
  AN2 AND2_24(G282,G317,G328);
  AN3 AND3_12(G117,G1,G39,G313);
  AN3 AND3_13(G277,G323,G281,G280);
  AN2 AND2_25(G278,G280,G42);
  AN3 AND3_14(G121,G318,G317,G328);
  AN3 AND3_15(G128,G280,G318,G40);
  AN2 AND2_26(G232,G38,G318);
  AN2 AND2_27(G233,G15,G318);
  AN2 AND2_28(G251,G318,G313);
  AN2 AND2_29(G252,G318,G317);
  AN4 AND4_3(G271,G318,G15,G14,G270);
  AN4 AND4_4(G210,G39,G38,G245,G209);
  AN2 AND2_30(G226,G318,G225);
  AN2 AND2_31(G175,G317,G176);
  AN4 AND4_5(G197,G8,G7,G6,G196);
  AN3 AND3_16(G263,G39,G38,G262);
  AN4 AND4_6(G150,G256,G147,G148,G149);
  AN2 AND2_32(G158,G280,G157);
  AN2 AND2_33(G185,G280,G184);
  AN4 AND4_7(G174,G41,G40,G15,G173);
  AN4 AND4_8(G211,G317,G39,G256,G212);
  AN2 AND2_34(G223,G16,G222);
  AN3 AND3_17(G272,G318,G4,G274);
  AN2 AND2_35(G264,G318,G266);
  AN2 AND2_36(G294,G16,G293);
  AN4 AND4_9(G152,G313,G317,G318,G154);
  AN4 AND4_10(G218,G2,G323,G216,G217);
  AN4 AND4_11(G151,G38,G16,G256,G153);
  AN3 AND3_18(G273,G40,G39,G275);
  AN3 AND3_19(G258,G318,G280,G257);
  AN2 AND2_37(G219,G318,G220);
  AN2 AND2_38(G259,G41,G260);
  AN2 AND2_39(G90,G89,G88);
  AN2 AND2_40(G93,G92,G91);
  AN2 AND2_41(G96,G95,G94);
  AN2 AND2_42(G99,G98,G97);
  AN2 AND2_43(G102,G101,G100);
  OR2 OR2_0(G126,G10,G11);
  OR2 OR2_1(G124,G11,G12);
  OR2 OR2_2(G125,G10,G12);
  OR3 OR3_0(G107,G41,G40,G1);
  OR2 OR2_3(G145,G16,G41);
  OR2 OR2_4(G243,G5,G41);
  OR2 OR2_5(G111,G15,G42);
  OR2 OR2_6(G144,G16,G42);
  OR3 OR3_1(G239,G40,G41,G42);
  OR2 OR2_7(G287,G42,G5);
  OR2 OR2_8(G115,G39,G42);
  OR3 OR3_2(G183,G38,G39,G41);
  OR3 OR3_3(G237,G16,G39,G40);
  OR2 OR2_9(G246,G4,G39);
  OR4 OR4_0(G113,G203,G202,G112,G198);
  OR4 OR4_1(G132,G171,G11,G12,G42);
  OR4 OR4_2(G133,G10,G172,G12,G42);
  OR4 OR4_3(G182,G14,G267,G38,G39);
  OR4 OR4_4(G238,G14,G267,G40,G42);
  OR2 OR2_10(G241,G256,G317);
  OR2 OR2_11(G136,G4,G281);
  OR2 OR2_12(G116,G39,G313);
  OR2 OR2_13(G286,G42,G313);
  OR2 OR2_14(G108,G328,G15);
  OR3 OR3_4(G109,G201,G267,G328);
  OR3 OR3_5(G240,G256,G313,G328);
  OR2 OR2_15(G242,G41,G328);
  OR2 OR2_16(G244,G281,G328);
  OR2 OR2_17(G110,G280,G42);
  OR2 OR2_18(G134,G280,G42);
  OR2 OR2_19(G135,G280,G40);
  OR3 OR3_6(G114,G267,G318,G328);
  OR3 OR3_7(G236,G318,G317,G328);
  OR2 OR2_20(G248,G245,G318);
  OR4 OR4_5(G321,G317,G318,G38,G319);
  OR2 OR2_21(G180,G41,G178);
  OR4 OR4_6(G78,G39,G4,G73,G74);
  OR4 OR4_7(G285,G3,G2,G1,G284);
  OR4 OR4_8(G63,G40,G318,G4,G59);
  OR4 OR4_9(G106,G8,G7,G203,G105);
  OR4 OR4_10(G308,G40,G318,G16,G304);
  OR4 OR4_11(G320,G40,G39,G38,G316);
  OR4 OR4_12(G52,G328,G313,G39,G50);
  OR2 OR2_22(G139,G317,G137);
  OR2 OR2_23(G255,G317,G253);
  OR4 OR4_13(G207,G202,G203,G204,G205);
  OR3 OR3_8(G309,G39,G38,G305);
  OR4 OR4_14(G62,G267,G4,G57,G58);
  OR4 OR4_15(G307,G328,G313,G39,G303);
  OR4 OR4_16(G85,G328,G313,G317,G81);
  OR3 OR3_9(G67,G174,G175,G177);
  OR4 OR4_17(G70,G318,G4,G65,G66);
  OR4 OR4_18(G89,G150,G151,G152,G155);
  OR4 OR4_19(G79,G40,G281,G4,G75);
  OR3 OR3_10(G64,G317,G318,G60);
  OR3 OR3_11(G72,G317,G318,G68);
  OR4 OR4_20(G71,G39,G281,G4,G67);
  OR2 OR2_24(G86,G38,G82);
  OR2 OR2_25(G80,G38,G76);
  OR2 OR2_26(G87,G281,G83);
  ND2 NAND2_0(G204,G9,G8);
  ND3 NAND3_0(G73,G42,G41,G40);
  ND2 NAND2_1(G319,G42,G41);
  ND4 NAND4_0(G123,G124,G125,G126,G256);
  ND3 NAND3_1(G65,G42,G41,G317);
  ND4 NAND4_1(G295,G41,G317,G39,G256);
  ND2 NAND2_2(G284,G42,G313);
  ND4 NAND4_2(G291,G313,G317,G39,G15);
  ND4 NAND4_3(G329,G313,G317,G39,G15);
  ND2 NAND2_3(G59,G144,G145);
  ND4 NAND4_4(G105,G328,G40,G15,G9);
  ND2 NAND2_4(G225,G41,G256);
  ND2 NAND2_5(G316,G328,G313);
  ND4 NAND4_5(G48,G40,G39,G280,G130);
  ND4 NAND4_6(G56,G40,G39,G280,G5);
  ND4 NAND4_7(G176,G42,G41,G280,G15);
  ND4 NAND4_8(G289,G313,G40,G39,G280);
  ND4 NAND4_9(G297,G41,G40,G39,G280);
  ND4 NAND4_10(G311,G313,G40,G39,G280);
  ND4 NAND4_11(G314,G40,G39,G280,G16);
  ND4 NAND4_12(G326,G313,G40,G39,G280);
  ND4 NAND4_13(G301,G281,G3,G323,G119);
  ND4 NAND4_14(G44,G317,G318,G280,G15);
  ND4 NAND4_15(G54,G41,G317,G318,G280);
  ND4 NAND4_16(G57,G41,G40,G318,G16);
  ND3 NAND3_2(G156,G318,G280,G281);
  ND4 NAND4_17(G299,G318,G280,G15,G14);
  ND2 NAND2_6(G262,G113,G317);
  ND2 NAND2_7(G179,G182,G183);
  ND2 NAND2_8(G205,G228,G229);
  ND4 NAND4_18(G224,G238,G239,G240,G241);
  ND4 NAND4_19(G227,G242,G243,G244,G40);
  ND4 NAND4_20(G266,G109,G110,G111,G40);
  ND4 NAND4_21(G293,G8,G7,G6,G131);
  ND3 NAND3_3(G58,G132,G133,G134);
  ND2 NAND2_9(G303,G135,G136);
  ND4 NAND4_22(G269,G114,G115,G116,G317);
  ND2 NAND2_10(G217,G236,G237);
  ND3 NAND3_4(G81,G246,G247,G248);
  ND4 NAND4_23(G46,G318,G280,G16,G122);
  ND4 NAND4_24(G69,G180,G328,G317,G179);
  ND3 NAND3_5(G275,G285,G286,G287);
  ND3 NAND3_6(G257,G106,G107,G108);
  ND2 NAND2_11(G315,G320,G321);
  ND2 NAND2_12(G306,G139,G138);
  ND2 NAND2_13(G84,G255,G254);
  ND2 NAND2_14(G49,G52,G51);
  ND4 NAND4_25(G61,G328,G313,G317,G146);
  ND2 NAND2_15(G75,G207,G206);
  ND4 NAND4_26(G302,G307,G308,G309,G306);
  ND4 NAND4_27(G92,G62,G63,G64,G61);
  ND4 NAND4_28(G95,G70,G71,G72,G69);
  ND4 NAND4_29(G98,G78,G79,G80,G77);
  ND4 NAND4_30(G101,G85,G86,G87,G84);
  NR2 NOR2_0(G216,G41,G3);
  NR2 NOR2_1(G140,G42,G41);
  NR2 NOR2_2(G119,G39,G38);
  NR4 NOR4_0(G178,G16,G3,G181,G1);
  NR3 NOR3_0(G74,G281,G267,G201);
  NR3 NOR3_1(G147,G38,G281,G267);
  NR4 NOR4_1(G148,G42,G313,G317,G39);
  NR3 NOR3_2(G270,G42,G313,G40);
  NR3 NOR3_3(G209,G328,G313,G317);
  NR2 NOR2_3(G304,G328,G313);
  NR2 NOR2_4(G50,G40,G280);
  NR3 NOR3_4(G131,G280,G267,G198);
  NR3 NOR3_5(G137,G42,G41,G280);
  NR2 NOR2_5(G177,G195,G280);
  NR3 NOR3_6(G196,G280,G267,G198);
  NR3 NOR3_7(G253,G42,G41,G280);
  NR2 NOR2_6(G138,G318,G256);
  NR2 NOR2_7(G254,G318,G256);
  NR2 NOR2_8(G122,G267,G123);
  NR2 NOR2_9(G149,G169,G170);
  NR2 NOR2_10(G165,G166,G167);
  NR2 NOR2_11(G192,G199,G200);
  NR2 NOR2_12(G290,G42,G291);
  NR2 NOR2_13(G327,G328,G329);
  NR3 NOR3_8(G305,G141,G142,G143);
  NR4 NOR4_2(G157,G160,G161,G162,G163);
  NR4 NOR4_3(G184,G187,G188,G189,G190);
  NR2 NOR2_14(G173,G193,G194);
  NR3 NOR3_9(G212,G213,G214,G215);
  NR2 NOR2_15(G222,G234,G235);
  NR2 NOR2_16(G274,G282,G283);
  NR3 NOR3_10(G47,G42,G41,G48);
  NR3 NOR3_11(G55,G42,G41,G56);
  NR2 NOR2_17(G104,G117,G118);
  NR4 NOR4_4(G154,G276,G277,G278,G279);
  NR2 NOR2_18(G288,G42,G289);
  NR2 NOR2_19(G296,G42,G297);
  NR2 NOR2_20(G310,G328,G311);
  NR3 NOR3_12(G312,G328,G313,G314);
  NR2 NOR2_21(G325,G328,G326);
  NR4 NOR4_5(G300,G42,G41,G40,G301);
  NR3 NOR3_13(G43,G42,G313,G44);
  NR2 NOR2_22(G53,G42,G54);
  NR2 NOR2_23(G324,G120,G121);
  NR3 NOR3_14(G51,G127,G128,G129);
  NR4 NOR4_6(G146,G3,G181,G1,G156);
  NR3 NOR3_15(G206,G231,G232,G233);
  NR4 NOR4_7(G153,G249,G250,G251,G252);
  NR4 NOR4_8(G298,G42,G313,G40,G299);
  NR2 NOR2_24(G159,G164,G165);
  NR2 NOR2_25(G186,G191,G192);
  NR2 NOR2_26(G221,G226,G227);
  NR4 NOR4_9(G155,G103,G328,G317,G104);
  NR2 NOR2_27(G66,G197,G281);
  NR2 NOR2_28(G261,G268,G269);
  NR4 NOR4_10(G322,G41,G38,G323,G324);
  NR4 NOR4_11(G45,G42,G313,G317,G46);
  NR2 NOR2_29(G60,G158,G159);
  NR2 NOR2_30(G68,G185,G186);
  NR2 NOR2_31(G77,G210,G211);
  NR2 NOR2_32(G220,G223,G224);
  NR3 NOR3_16(G260,G263,G264,G265);
  NR3 NOR3_17(G292,G294,G328,G295);
  NR3 NOR3_18(G82,G271,G272,G273);
  NR3 NOR3_19(G76,G218,G219,G221);
  NR3 NOR3_20(G83,G258,G259,G261);

endmodule