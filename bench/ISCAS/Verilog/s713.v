module s713(VDD,CK,G1,G10,G100BF,G101BF,G103BF,G104BF,G105BF,G106BF,G107,
  G11,G12,G13,
  G14,G15,G16,G17,G18,G19,G2,G20,G21,G22,G23,G24,G25,G26,G27,G28,G29,G3,G30,
  G31,G32,G33,G34,G35,G36,G4,G5,G6,G8,G83,G84,G85,G86BF,G87BF,G88BF,G89BF,G9,
  G90,G91,G92,G94,G95BF,G96BF,G97BF,G98BF,G99BF);
input VDD,CK,G1,G2,G3,G4,G5,G6,G8,G9,G10,G11,G12,G13,G14,G15,G16,G17,G18,
  G19,G20,G21,
  G22,G23,G24,G25,G26,G27,G28,G29,G30,G31,G32,G33,G34,G35,G36;
output G103BF,G104BF,G105BF,G106BF,G107,G83,G84,G85,G86BF,G87BF,G88BF,G89BF,
  G90,G91,G92,G94,G95BF,G96BF,G97BF,G98BF,G99BF,G100BF,G101BF;

  wire G64,G380,G65,G262,G66,G394,G67,G250,G68,G122,G69,G133,G70,G138,G71,G139,
    G72,G140,G73,G141,G74,G142,G75,G125,G76,G126,G77,G127,G78,G128,G79,G129,
    G80,G130,G81,G131,G82,G132,IIII633,G366,G379,IIII643,IIII646,IIII649,
    IIII652,IIII655,IIII660,IIII680,IIII684,IIII687,II165,IIII178,II169,G113,
    II172,G115,II175,G117,II178,G219,II181,G119,II184,G221,II187,G121,II190,
    G223,II193,G209,II196,G109,II199,G211,II202,G111,II205,G213,II208,G215,
    II211,G217,G352,G360,G361,G362,G363,G364,G367,G386,G388,G389,G110,G114,
    G118,G216,G218,G220,G222,G365,G368,G387,G225,G390,IIII356,G289,II254,G324,
    G166,II257,G325,II260,G338,G194,II263,G339,II266,G344,G202,II269,G345,
    II272,G312,G313,II275,G315,G316,II278,G318,G319,II281,G321,G322,G143,II287,
    G381,II291,G375,II295,G371,II303,G350,IIII299,G281,IIII313,G283,G382,G100,
    G376,G98,G372,G96,IIII301,IIII315,II321,G135,G329,II324,G137,G333,G87,
    IIII406,G89,IIII422,G173,G183,II335,G174,II338,G184,II341,G355,G359,G356,
    G108,G116,II354,G293,G146,II357,G294,II360,G309,G162,II363,G310,II366,G341,
    G198,II369,G342,II372,G303,G154,II375,G304,II378,G383,II382,G396,II386,
    G373,II390,G392,G384,G101,G397,G106,G374,G97,G393,G104,IIII476,IIII279,
    G278,G224,IIII306,G282,IIII334,G286,IIII327,G285,IIII208,G268,IIII308,
    IIII336,IIII329,IIII210,II442,G136,G331,G88,IIII414,G178,II449,G179,II452,
    G357,G358,G112,II460,G335,G190,II463,G336,II466,G306,G158,II469,G307,II472,
    G377,II476,G378,G99,G395,G105,IIII272,G277,IIII265,G276,IIII320,G284,
    IIII285,G279,IIII292,G280,IIII322,IIII287,IIII294,II517,G134,G327,G86,
    IIII398,G168,II524,G169,II527,G353,G354,G120,II535,G347,G206,II538,G348,
    II541,G300,G150,II544,G301,II547,G369,II551,G370,G95,G391,G103,IIII230,
    G271,IIII258,G275,IIII348,G288,IIII341,G287,IIII222,G270,IIII350,IIII343,
    IIII237,G272,IIII244,G273,IIII251,G274,IIII224,II608,G124,G298,G231,G232,
    G233,G234,G247,G248,G263,G264,G214,G210,G240,G266,G229,G245,G253,IIII533,
    G227,G243,G249,G265,G236,G237,G252,IIII527,G212,G251,IIII512,IIII538,G228,
    G244,G256,G230,G235,G246,IIII515,G261,G208,IIII495,G255,G257,IIII537,G226,
    G242,IIII553,G241,G267,G238,G239,G254,IIII518,IIII521,IIII524,G258,G259,
    G260,IIII546,IIII300,IIII314,IIII307,IIII335,IIII328,IIII209,IIII321,
    IIII286,IIII293,IIII349,IIII342,IIII223;

  FD1 DFF_0(CK,G64,G380);
  FD1 DFF_1(CK,G65,G262);
  FD1 DFF_2(CK,G66,G394);
  FD1 DFF_3(CK,G67,G250);
  FD1 DFF_4(CK,G68,G122);
  FD1 DFF_5(CK,G69,G133);
  FD1 DFF_6(CK,G70,G138);
  FD1 DFF_7(CK,G71,G139);
  FD1 DFF_8(CK,G72,G140);
  FD1 DFF_9(CK,G73,G141);
  FD1 DFF_10(CK,G74,G142);
  FD1 DFF_11(CK,G75,G125);
  FD1 DFF_12(CK,G76,G126);
  FD1 DFF_13(CK,G77,G127);
  FD1 DFF_14(CK,G78,G128);
  FD1 DFF_15(CK,G79,G129);
  FD1 DFF_16(CK,G80,G130);
  FD1 DFF_17(CK,G81,G131);
  FD1 DFF_18(CK,G82,G132);
  IV  NOT_0(IIII633,G1);
  IV  NOT_1(G366,G2);
  IV  NOT_2(G379,G3);
  IV  NOT_3(IIII643,G4);
  IV  NOT_4(IIII646,G5);
  IV  NOT_5(IIII649,G6);
  IV  NOT_6(IIII652,G8);
  IV  NOT_7(IIII655,G9);
  IV  NOT_8(IIII660,G10);
  IV  NOT_9(IIII680,G11);
  IV  NOT_10(IIII684,G12);
  IV  NOT_11(IIII687,G13);
  IV  NOT_12(II165,G27);
  IV  NOT_13(G91,II165);
  IV  NOT_14(IIII178,G29);
  IV  NOT_15(II169,G70);
  IV  NOT_16(G113,II169);
  IV  NOT_17(II172,G71);
  IV  NOT_18(G115,II172);
  IV  NOT_19(II175,G72);
  IV  NOT_20(G117,II175);
  IV  NOT_21(II178,G80);
  IV  NOT_22(G219,II178);
  IV  NOT_23(II181,G73);
  IV  NOT_24(G119,II181);
  IV  NOT_25(II184,G81);
  IV  NOT_26(G221,II184);
  IV  NOT_27(II187,G74);
  IV  NOT_28(G121,II187);
  IV  NOT_29(II190,G82);
  IV  NOT_30(G223,II190);
  IV  NOT_31(II193,G75);
  IV  NOT_32(G209,II193);
  IV  NOT_33(II196,G68);
  IV  NOT_34(G109,II196);
  IV  NOT_35(II199,G76);
  IV  NOT_36(G211,II199);
  IV  NOT_37(II202,G69);
  IV  NOT_38(G111,II202);
  IV  NOT_39(II205,G77);
  IV  NOT_40(G213,II205);
  IV  NOT_41(II208,G78);
  IV  NOT_42(G215,II208);
  IV  NOT_43(II211,G79);
  IV  NOT_44(G217,II211);
  IV  NOT_45(G352,IIII633);
  IV  NOT_46(G360,IIII643);
  IV  NOT_47(G361,IIII646);
  IV  NOT_48(G362,IIII649);
  IV  NOT_49(G363,IIII652);
  IV  NOT_50(G364,IIII655);
  IV  NOT_51(G367,IIII660);
  IV  NOT_52(G386,IIII680);
  IV  NOT_53(G388,IIII684);
  IV  NOT_54(G389,IIII687);
  IV  NOT_55(G94,IIII178);
  IV  NOT_56(G110,G360);
  IV  NOT_57(G114,G360);
  IV  NOT_58(G118,G360);
  IV  NOT_59(G216,G360);
  IV  NOT_60(G218,G360);
  IV  NOT_61(G220,G360);
  IV  NOT_62(G222,G360);
  IV  NOT_63(G365,G364);
  IV  NOT_64(G368,G367);
  IV  NOT_65(G387,G386);
  IV  NOT_66(G225,G388);
  IV  NOT_67(G390,G389);
  IV  NOT_68(IIII356,G289);
  IV  NOT_69(II254,G324);
  IV  NOT_70(G166,II254);
  IV  NOT_71(II257,G324);
  IV  NOT_72(G325,II257);
  IV  NOT_73(II260,G338);
  IV  NOT_74(G194,II260);
  IV  NOT_75(II263,G338);
  IV  NOT_76(G339,II263);
  IV  NOT_77(II266,G344);
  IV  NOT_78(G202,II266);
  IV  NOT_79(II269,G344);
  IV  NOT_80(G345,II269);
  IV  NOT_81(II272,G312);
  IV  NOT_82(G313,II272);
  IV  NOT_83(II275,G315);
  IV  NOT_84(G316,II275);
  IV  NOT_85(II278,G318);
  IV  NOT_86(G319,II278);
  IV  NOT_87(II281,G321);
  IV  NOT_88(G322,II281);
  IV  NOT_89(G143,IIII356);
  IV  NOT_90(II287,G166);
  IV  NOT_91(G381,II287);
  IV  NOT_92(II291,G194);
  IV  NOT_93(G375,II291);
  IV  NOT_94(II295,G202);
  IV  NOT_95(G371,II295);
  IV  NOT_96(II303,G143);
  IV  NOT_97(G350,II303);
  IV  NOT_98(IIII299,G281);
  IV  NOT_99(IIII313,G283);
  IV  NOT_100(G382,G381);
  IV  NOT_101(G100BF,G100);
  IV  NOT_102(G376,G375);
  IV  NOT_103(G98BF,G98);
  IV  NOT_104(G372,G371);
  IV  NOT_105(G96BF,G96);
  IV  NOT_106(IIII301,IIII299);
  IV  NOT_107(IIII315,IIII313);
  IV  NOT_108(II321,G135);
  IV  NOT_109(G329,II321);
  IV  NOT_110(II324,G137);
  IV  NOT_111(G333,II324);
  IV  NOT_112(G87BF,G87);
  IV  NOT_113(IIII406,G87);
  IV  NOT_114(G89BF,G89);
  IV  NOT_115(IIII422,G89);
  IV  NOT_116(G173,IIII406);
  IV  NOT_117(G183,IIII422);
  IV  NOT_118(II335,G173);
  IV  NOT_119(G174,II335);
  IV  NOT_120(II338,G183);
  IV  NOT_121(G184,II338);
  IV  NOT_122(II341,G174);
  IV  NOT_123(G355,II341);
  IV  NOT_124(G359,G184);
  IV  NOT_125(G356,G355);
  IV  NOT_126(G108,G359);
  IV  NOT_127(G116,G356);
  IV  NOT_128(II354,G293);
  IV  NOT_129(G146,II354);
  IV  NOT_130(II357,G293);
  IV  NOT_131(G294,II357);
  IV  NOT_132(II360,G309);
  IV  NOT_133(G162,II360);
  IV  NOT_134(II363,G309);
  IV  NOT_135(G310,II363);
  IV  NOT_136(II366,G341);
  IV  NOT_137(G198,II366);
  IV  NOT_138(II369,G341);
  IV  NOT_139(G342,II369);
  IV  NOT_140(II372,G303);
  IV  NOT_141(G154,II372);
  IV  NOT_142(II375,G303);
  IV  NOT_143(G304,II375);
  IV  NOT_144(II378,G146);
  IV  NOT_145(G383,II378);
  IV  NOT_146(II382,G162);
  IV  NOT_147(G396,II382);
  IV  NOT_148(II386,G198);
  IV  NOT_149(G373,II386);
  IV  NOT_150(II390,G154);
  IV  NOT_151(G392,II390);
  IV  NOT_152(G384,G383);
  IV  NOT_153(G101BF,G101);
  IV  NOT_154(G397,G396);
  IV  NOT_155(G106BF,G106);
  IV  NOT_156(G374,G373);
  IV  NOT_157(G97BF,G97);
  IV  NOT_158(G393,G392);
  IV  NOT_159(G104BF,G104);
  IV  NOT_160(IIII476,G384);
  IV  NOT_161(IIII279,G278);
  IV  NOT_162(G224,IIII476);
  IV  NOT_163(G132,IIII279);
  IV  NOT_164(IIII306,G282);
  IV  NOT_165(IIII334,G286);
  IV  NOT_166(IIII327,G285);
  IV  NOT_167(IIII208,G268);
  IV  NOT_168(IIII308,IIII306);
  IV  NOT_169(IIII336,IIII334);
  IV  NOT_170(IIII329,IIII327);
  IV  NOT_171(IIII210,IIII208);
  IV  NOT_172(II442,G136);
  IV  NOT_173(G331,II442);
  IV  NOT_174(G88BF,G88);
  IV  NOT_175(IIII414,G88);
  IV  NOT_176(G178,IIII414);
  IV  NOT_177(II449,G178);
  IV  NOT_178(G179,II449);
  IV  NOT_179(II452,G179);
  IV  NOT_180(G357,II452);
  IV  NOT_181(G358,G357);
  IV  NOT_182(G112,G358);
  IV  NOT_183(II460,G335);
  IV  NOT_184(G190,II460);
  IV  NOT_185(II463,G335);
  IV  NOT_186(G336,II463);
  IV  NOT_187(II466,G306);
  IV  NOT_188(G158,II466);
  IV  NOT_189(II469,G306);
  IV  NOT_190(G307,II469);
  IV  NOT_191(II472,G190);
  IV  NOT_192(G377,II472);
  IV  NOT_193(II476,G158);
  IV  NOT_194(G394,II476);
  IV  NOT_195(G378,G377);
  IV  NOT_196(G99BF,G99);
  IV  NOT_197(G395,G158);
  IV  NOT_198(G105BF,G105);
  IV  NOT_199(IIII272,G277);
  IV  NOT_200(G131,IIII272);
  IV  NOT_201(IIII265,G276);
  IV  NOT_202(IIII320,G284);
  IV  NOT_203(IIII285,G279);
  IV  NOT_204(IIII292,G280);
  IV  NOT_205(G130,IIII265);
  IV  NOT_206(IIII322,IIII320);
  IV  NOT_207(IIII287,IIII285);
  IV  NOT_208(IIII294,IIII292);
  IV  NOT_209(II517,G134);
  IV  NOT_210(G327,II517);
  IV  NOT_211(G86BF,G86);
  IV  NOT_212(IIII398,G86);
  IV  NOT_213(G168,IIII398);
  IV  NOT_214(II524,G168);
  IV  NOT_215(G169,II524);
  IV  NOT_216(II527,G169);
  IV  NOT_217(G353,II527);
  IV  NOT_218(G354,G353);
  IV  NOT_219(G120,G354);
  IV  NOT_220(II535,G347);
  IV  NOT_221(G206,II535);
  IV  NOT_222(II538,G347);
  IV  NOT_223(G348,II538);
  IV  NOT_224(II541,G300);
  IV  NOT_225(G150,II541);
  IV  NOT_226(II544,G300);
  IV  NOT_227(G301,II544);
  IV  NOT_228(II547,G206);
  IV  NOT_229(G369,II547);
  IV  NOT_230(II551,G150);
  IV  NOT_231(G380,II551);
  IV  NOT_232(G370,G369);
  IV  NOT_233(G95BF,G95);
  IV  NOT_234(G391,G150);
  IV  NOT_235(G103BF,G103);
  IV  NOT_236(IIII230,G271);
  IV  NOT_237(IIII258,G275);
  IV  NOT_238(IIII348,G288);
  IV  NOT_239(IIII341,G287);
  IV  NOT_240(G125,IIII230);
  IV  NOT_241(G129,IIII258);
  IV  NOT_242(IIII222,G270);
  IV  NOT_243(IIII350,IIII348);
  IV  NOT_244(IIII343,IIII341);
  IV  NOT_245(IIII237,G272);
  IV  NOT_246(IIII244,G273);
  IV  NOT_247(IIII251,G274);
  IV  NOT_248(IIII224,IIII222);
  IV  NOT_249(G126,IIII237);
  IV  NOT_250(G127,IIII244);
  IV  NOT_251(G128,IIII251);
  IV  NOT_252(II608,G124);
  IV  NOT_253(G298,II608);
  AN3 AND3_0(G289,G386,G388,G389);
  AN2 AND2_0(G324,G110,G111);
  AN2 AND2_1(G338,G114,G115);
  AN2 AND2_2(G344,G118,G119);
  AN2 AND2_3(G312,G216,G217);
  AN2 AND2_4(G315,G218,G219);
  AN2 AND2_5(G318,G220,G221);
  AN2 AND2_6(G321,G222,G223);
  AN2 AND2_7(G231,G379,G387);
  AN2 AND2_8(G232,G379,G387);
  AN2 AND2_9(G233,G379,G387);
  AN2 AND2_10(G234,G379,G387);
  AN4 AND4_0(G247,G379,G365,G368,G390);
  AN4 AND4_1(G248,G379,G365,G367,G390);
  AN4 AND4_2(G263,G379,G364,G368,G390);
  AN4 AND4_3(G264,G379,G364,G367,G390);
  AN2 AND2_11(G100,G325,G35);
  AN2 AND2_12(G98,G339,G33);
  AN2 AND2_13(G96,G345,G31);
  AN2 AND2_14(G107,G313,G18);
  AN2 AND2_15(G83,G316,G19);
  AN2 AND2_16(G84,G319,G20);
  AN2 AND2_17(G85,G322,G21);
  AN2 AND2_18(G92,G350,G28);
  AN2 AND2_19(G87,G329,G23);
  AN2 AND2_20(G89,G333,G25);
  AN2 AND2_21(G293,G108,G109);
  AN2 AND2_22(G309,G214,G215);
  AN2 AND2_23(G341,G116,G117);
  AN2 AND2_24(G303,G210,G211);
  AN2 AND2_25(G101,G294,G36);
  AN2 AND2_26(G106,G310,G17);
  AN2 AND2_27(G97,G342,G32);
  AN2 AND2_28(G104,G304,G15);
  AN2 AND2_29(G240,G359,G383);
  AN4 AND4_4(G266,G364,G367,G383,G390);
  AN2 AND2_30(G229,G366,G396);
  AN2 AND2_31(G245,G352,G396);
  AN2 AND2_32(G250,G366,G396);
  AN2 AND2_33(G278,G366,G396);
  AN3 AND3_1(G253,G356,G373,G375);
  AN3 AND3_2(IIII533,G365,G367,G373);
  AN2 AND2_34(G227,G366,G392);
  AN2 AND2_35(G243,G392,G361);
  AN3 AND3_3(G249,G366,G66,G397);
  AN3 AND3_4(G265,G375,G390,IIII533);
  AN2 AND2_36(G236,G374,G376);
  AN2 AND2_37(G237,G374,G375);
  AN3 AND3_5(G252,G355,G374,G375);
  AN3 AND3_6(IIII527,G366,G64,G393);
  AN2 AND2_38(G88,G331,G24);
  AN2 AND2_39(G335,G112,G113);
  AN2 AND2_40(G306,G212,G213);
  AN2 AND2_41(G99,G336,G34);
  AN2 AND2_42(G105,G307,G16);
  AN3 AND3_7(G251,G358,G377,G381);
  AN3 AND3_8(IIII512,G364,G368,G377);
  AN4 AND4_5(IIII538,G377,G381,G383,G387);
  AN2 AND2_43(G228,G366,G158);
  AN2 AND2_44(G244,G158,G362);
  AN3 AND3_9(G277,G366,G158,G397);
  AN3 AND3_10(G256,G381,G390,IIII512);
  AN2 AND2_45(G230,G378,G382);
  AN2 AND2_46(G235,G378,G381);
  AN3 AND3_11(G246,G357,G378,G381);
  AN3 AND3_12(IIII515,G393,G395,G397);
  AN3 AND3_13(G261,G395,G397,IIII527);
  AN4 AND4_6(G262,G366,G392,G395,G397);
  AN4 AND4_7(G276,G366,G392,G395,G397);
  AN2 AND2_47(G86,G327,G22);
  AN2 AND2_48(G347,G120,G121);
  AN2 AND2_49(G300,G208,G209);
  AN2 AND2_50(G95,G348,G30);
  AN2 AND2_51(G103,G301,G14);
  AN3 AND3_14(IIII495,G365,G368,G369);
  AN3 AND3_15(G255,G354,G369,G371);
  AN4 AND4_8(G257,G363,G369,G371,IIII515);
  AN4 AND4_9(IIII537,G369,G371,G373,G375);
  AN2 AND2_52(G226,G366,G150);
  AN2 AND2_53(G242,G150,G363);
  AN3 AND3_16(IIII553,G366,G150,G393);
  AN3 AND3_17(G241,G371,G390,IIII495);
  AN2 AND2_54(G267,IIII537,IIII538);
  AN2 AND2_55(G238,G370,G372);
  AN2 AND2_56(G239,G370,G371);
  AN3 AND3_18(G254,G353,G370,G371);
  AN3 AND3_19(G275,G395,G397,IIII553);
  AN3 AND3_20(IIII518,G391,G395,G397);
  AN3 AND3_21(IIII521,G391,G393,G397);
  AN3 AND3_22(IIII524,G352,G391,G393);
  AN4 AND4_10(G258,G361,G373,G375,IIII518);
  AN4 AND4_11(G259,G362,G377,G381,IIII521);
  AN3 AND3_23(G260,G395,G383,IIII524);
  AN2 AND2_57(G90,G298,G26);
  OR3 OR3_0(G281,G232,G248,G65);
  OR3 OR3_1(G283,G234,G67,G264);
  OR3 OR3_2(G282,G233,G249,G263);
  OR2 OR2_0(G286,G237,G253);
  OR2 OR2_1(G285,G236,G252);
  OR2 OR2_2(G268,G224,G240);
  OR2 OR2_3(G284,G235,G251);
  OR2 OR2_4(G279,G230,G246);
  OR3 OR3_3(G280,G231,G247,G261);
  OR3 OR3_4(G271,G226,G242,G257);
  OR3 OR3_5(IIII546,G225,G241,G256);
  OR2 OR2_5(G288,G239,G255);
  OR2 OR2_6(G287,G238,G254);
  OR4 OR4_0(G270,G265,G266,G267,IIII546);
  OR3 OR3_6(G272,G227,G243,G258);
  OR3 OR3_7(G273,G228,G244,G259);
  OR3 OR3_8(G274,G229,G245,G260);
  ND2 NAND2_0(IIII300,G281,IIII299);
  ND2 NAND2_1(IIII314,G283,IIII313);
  ND2 NAND2_2(G135,IIII300,IIII301);
  ND2 NAND2_3(G137,IIII314,IIII315);
  ND2 NAND2_4(G214,G379,G359);
  ND2 NAND2_5(G210,G379,G356);
  ND2 NAND2_6(IIII307,G282,IIII306);
  ND2 NAND2_7(IIII335,G286,IIII334);
  ND2 NAND2_8(IIII328,G285,IIII327);
  ND2 NAND2_9(IIII209,G268,IIII208);
  ND2 NAND2_10(G136,IIII307,IIII308);
  ND2 NAND2_11(G140,IIII335,IIII336);
  ND2 NAND2_12(G139,IIII328,IIII329);
  ND2 NAND2_13(G122,IIII209,IIII210);
  ND2 NAND2_14(G212,G379,G358);
  ND2 NAND2_15(IIII321,G284,IIII320);
  ND2 NAND2_16(IIII286,G279,IIII285);
  ND2 NAND2_17(IIII293,G280,IIII292);
  ND2 NAND2_18(G138,IIII321,IIII322);
  ND2 NAND2_19(G133,IIII286,IIII287);
  ND2 NAND2_20(G134,IIII293,IIII294);
  ND2 NAND2_21(G208,G379,G354);
  ND2 NAND2_22(IIII349,G288,IIII348);
  ND2 NAND2_23(IIII342,G287,IIII341);
  ND2 NAND2_24(IIII223,G270,IIII222);
  ND2 NAND2_25(G142,IIII349,IIII350);
  ND2 NAND2_26(G141,IIII342,IIII343);
  ND2 NAND2_27(G124,IIII223,IIII224);

endmodule
