// C1355.blif
module foobar(p0, q0, r0, s0, t0, u0, v0, w0, x0, y0, z0, a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1, q1, r1, s1, t1, u1, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0);
input a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0;
output p0, q0, r0, s0, t0, u0, v0, w0, x0, y0, z0, a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1, q1, r1, s1, t1, u1;
and(v1, n0, o0);
and(w1, m0, o0);
and(x1, l0, o0);
and(y1, k0, o0);
and(z1, j0, o0);
and(a2, i0, o0);
and(b2, h0, o0);
and(c2, g0, o0);
not(t_0, f0);
not(t_1, e0);
or(d2, t_0, t_1);
not(t_2, d0);
not(t_3, c0);
or(e2, t_2, t_3);
not(t_4, f0);
not(t_5, b0);
or(f2, t_4, t_5);
not(t_6, e0);
not(t_7, a0);
or(g2, t_6, t_7);
not(t_8, b0);
not(t_9, a0);
or(h2, t_8, t_9);
not(t_10, d0);
not(t_11, z);
or(i2, t_10, t_11);
not(t_12, c0);
not(t_13, y);
or(j2, t_12, t_13);
not(t_14, z);
not(t_15, y);
or(k2, t_14, t_15);
not(t_16, x);
not(t_17, w);
or(l2, t_16, t_17);
not(t_18, v);
not(t_19, u);
or(m2, t_18, t_19);
not(t_20, x);
not(t_21, t);
or(n2, t_20, t_21);
not(t_22, w);
not(t_23, s);
or(o2, t_22, t_23);
not(t_24, t);
not(t_25, s);
or(p2, t_24, t_25);
not(t_26, v);
not(t_27, r);
or(q2, t_26, t_27);
not(t_28, u);
not(t_29, q);
or(r2, t_28, t_29);
not(t_30, r);
not(t_31, q);
or(s2, t_30, t_31);
not(t_32, p);
not(t_33, o);
or(t2, t_32, t_33);
not(t_34, n);
not(t_35, m);
or(u2, t_34, t_35);
not(t_36, p);
not(t_37, l);
or(v2, t_36, t_37);
not(t_38, o);
not(t_39, k);
or(w2, t_38, t_39);
not(t_40, l);
not(t_41, k);
or(x2, t_40, t_41);
not(t_42, n);
not(t_43, j);
or(y2, t_42, t_43);
not(t_44, m);
not(t_45, i);
or(z2, t_44, t_45);
not(t_46, j);
not(t_47, i);
or(a3, t_46, t_47);
not(t_48, h);
not(t_49, g);
or(b3, t_48, t_49);
not(t_50, f);
not(t_51, e);
or(c3, t_50, t_51);
not(t_52, h);
not(t_53, d);
or(d3, t_52, t_53);
not(t_54, g);
not(t_55, c);
or(e3, t_54, t_55);
not(t_56, d);
not(t_57, c);
or(f3, t_56, t_57);
not(t_58, f);
not(t_59, b);
or(g3, t_58, t_59);
not(t_60, e);
not(t_61, a);
or(h3, t_60, t_61);
not(t_62, b);
not(t_63, a);
or(i3, t_62, t_63);
not(t_64, f2);
not(t_65, f0);
or(j3, t_64, t_65);
not(t_66, d2);
not(t_67, f0);
or(k3, t_66, t_67);
not(t_68, g2);
not(t_69, e0);
or(l3, t_68, t_69);
not(t_70, d2);
not(t_71, e0);
or(m3, t_70, t_71);
not(t_72, i2);
not(t_73, d0);
or(n3, t_72, t_73);
not(t_74, e2);
not(t_75, d0);
or(o3, t_74, t_75);
not(t_76, j2);
not(t_77, c0);
or(p3, t_76, t_77);
not(t_78, e2);
not(t_79, c0);
or(q3, t_78, t_79);
not(t_80, f2);
not(t_81, b0);
or(r3, t_80, t_81);
not(t_82, h2);
not(t_83, b0);
or(s3, t_82, t_83);
not(t_84, g2);
not(t_85, a0);
or(t3, t_84, t_85);
not(t_86, h2);
not(t_87, a0);
or(u3, t_86, t_87);
not(t_88, i2);
not(t_89, z);
or(v3, t_88, t_89);
not(t_90, k2);
not(t_91, z);
or(w3, t_90, t_91);
not(t_92, j2);
not(t_93, y);
or(x3, t_92, t_93);
not(t_94, k2);
not(t_95, y);
or(y3, t_94, t_95);
not(t_96, n2);
not(t_97, x);
or(z3, t_96, t_97);
not(t_98, l2);
not(t_99, x);
or(a4, t_98, t_99);
not(t_100, o2);
not(t_101, w);
or(b4, t_100, t_101);
not(t_102, l2);
not(t_103, w);
or(c4, t_102, t_103);
not(t_104, q2);
not(t_105, v);
or(d4, t_104, t_105);
not(t_106, m2);
not(t_107, v);
or(e4, t_106, t_107);
not(t_108, r2);
not(t_109, u);
or(f4, t_108, t_109);
not(t_110, m2);
not(t_111, u);
or(g4, t_110, t_111);
not(t_112, n2);
not(t_113, t);
or(h4, t_112, t_113);
not(t_114, p2);
not(t_115, t);
or(i4, t_114, t_115);
not(t_116, o2);
not(t_117, s);
or(j4, t_116, t_117);
not(t_118, p2);
not(t_119, s);
or(k4, t_118, t_119);
not(t_120, q2);
not(t_121, r);
or(l4, t_120, t_121);
not(t_122, s2);
not(t_123, r);
or(m4, t_122, t_123);
not(t_124, r2);
not(t_125, q);
or(n4, t_124, t_125);
not(t_126, s2);
not(t_127, q);
or(o4, t_126, t_127);
not(t_128, v2);
not(t_129, p);
or(p4, t_128, t_129);
not(t_130, t2);
not(t_131, p);
or(q4, t_130, t_131);
not(t_132, w2);
not(t_133, o);
or(r4, t_132, t_133);
not(t_134, t2);
not(t_135, o);
or(s4, t_134, t_135);
not(t_136, y2);
not(t_137, n);
or(t4, t_136, t_137);
not(t_138, u2);
not(t_139, n);
or(u4, t_138, t_139);
not(t_140, z2);
not(t_141, m);
or(v4, t_140, t_141);
not(t_142, u2);
not(t_143, m);
or(w4, t_142, t_143);
not(t_144, v2);
not(t_145, l);
or(x4, t_144, t_145);
not(t_146, x2);
not(t_147, l);
or(y4, t_146, t_147);
not(t_148, w2);
not(t_149, k);
or(z4, t_148, t_149);
not(t_150, x2);
not(t_151, k);
or(a5, t_150, t_151);
not(t_152, y2);
not(t_153, j);
or(b5, t_152, t_153);
not(t_154, a3);
not(t_155, j);
or(c5, t_154, t_155);
not(t_156, z2);
not(t_157, i);
or(d5, t_156, t_157);
not(t_158, a3);
not(t_159, i);
or(e5, t_158, t_159);
not(t_160, d3);
not(t_161, h);
or(f5, t_160, t_161);
not(t_162, b3);
not(t_163, h);
or(g5, t_162, t_163);
not(t_164, e3);
not(t_165, g);
or(h5, t_164, t_165);
not(t_166, b3);
not(t_167, g);
or(i5, t_166, t_167);
not(t_168, g3);
not(t_169, f);
or(j5, t_168, t_169);
not(t_170, c3);
not(t_171, f);
or(k5, t_170, t_171);
not(t_172, h3);
not(t_173, e);
or(l5, t_172, t_173);
not(t_174, c3);
not(t_175, e);
or(m5, t_174, t_175);
not(t_176, d3);
not(t_177, d);
or(n5, t_176, t_177);
not(t_178, f3);
not(t_179, d);
or(o5, t_178, t_179);
not(t_180, e3);
not(t_181, c);
or(p5, t_180, t_181);
not(t_182, f3);
not(t_183, c);
or(q5, t_182, t_183);
not(t_184, g3);
not(t_185, b);
or(r5, t_184, t_185);
not(t_186, i3);
not(t_187, b);
or(s5, t_186, t_187);
not(t_188, h3);
not(t_189, a);
or(t5, t_188, t_189);
not(t_190, i3);
not(t_191, a);
or(u5, t_190, t_191);
not(t_192, j3);
not(t_193, r3);
or(v5, t_192, t_193);
not(t_194, k3);
not(t_195, m3);
or(w5, t_194, t_195);
not(t_196, l3);
not(t_197, t3);
or(x5, t_196, t_197);
not(t_198, n3);
not(t_199, v3);
or(y5, t_198, t_199);
not(t_200, o3);
not(t_201, q3);
or(z5, t_200, t_201);
not(t_202, p3);
not(t_203, x3);
or(a6, t_202, t_203);
not(t_204, s3);
not(t_205, u3);
or(b6, t_204, t_205);
not(t_206, w3);
not(t_207, y3);
or(c6, t_206, t_207);
not(t_208, z3);
not(t_209, h4);
or(d6, t_208, t_209);
not(t_210, a4);
not(t_211, c4);
or(e6, t_210, t_211);
not(t_212, b4);
not(t_213, j4);
or(f6, t_212, t_213);
not(t_214, d4);
not(t_215, l4);
or(g6, t_214, t_215);
not(t_216, e4);
not(t_217, g4);
or(h6, t_216, t_217);
not(t_218, f4);
not(t_219, n4);
or(i6, t_218, t_219);
not(t_220, i4);
not(t_221, k4);
or(j6, t_220, t_221);
not(t_222, m4);
not(t_223, o4);
or(k6, t_222, t_223);
not(t_224, p4);
not(t_225, x4);
or(l6, t_224, t_225);
not(t_226, q4);
not(t_227, s4);
or(m6, t_226, t_227);
not(t_228, r4);
not(t_229, z4);
or(n6, t_228, t_229);
not(t_230, t4);
not(t_231, b5);
or(o6, t_230, t_231);
not(t_232, u4);
not(t_233, w4);
or(p6, t_232, t_233);
not(t_234, v4);
not(t_235, d5);
or(q6, t_234, t_235);
not(t_236, y4);
not(t_237, a5);
or(r6, t_236, t_237);
not(t_238, c5);
not(t_239, e5);
or(s6, t_238, t_239);
not(t_240, f5);
not(t_241, n5);
or(t6, t_240, t_241);
not(t_242, g5);
not(t_243, i5);
or(u6, t_242, t_243);
not(t_244, h5);
not(t_245, p5);
or(v6, t_244, t_245);
not(t_246, j5);
not(t_247, r5);
or(w6, t_246, t_247);
not(t_248, k5);
not(t_249, m5);
or(x6, t_248, t_249);
not(t_250, l5);
not(t_251, t5);
or(y6, t_250, t_251);
not(t_252, o5);
not(t_253, q5);
or(z6, t_252, t_253);
not(t_254, s5);
not(t_255, u5);
or(a7, t_254, t_255);
not(t_256, v5);
not(t_257, d6);
or(b7, t_256, t_257);
not(t_258, w5);
not(t_259, z5);
or(c7, t_258, t_259);
not(t_260, x5);
not(t_261, f6);
or(d7, t_260, t_261);
not(t_262, y5);
not(t_263, g6);
or(e7, t_262, t_263);
not(t_264, a6);
not(t_265, i6);
or(f7, t_264, t_265);
not(t_266, b6);
not(t_267, c6);
or(g7, t_266, t_267);
not(t_268, e6);
not(t_269, h6);
or(h7, t_268, t_269);
not(t_270, j6);
not(t_271, k6);
or(i7, t_270, t_271);
not(t_272, l6);
not(t_273, t6);
or(j7, t_272, t_273);
not(t_274, m6);
not(t_275, p6);
or(k7, t_274, t_275);
not(t_276, n6);
not(t_277, v6);
or(l7, t_276, t_277);
not(t_278, o6);
not(t_279, w6);
or(m7, t_278, t_279);
not(t_280, q6);
not(t_281, y6);
or(n7, t_280, t_281);
not(t_282, r6);
not(t_283, s6);
or(o7, t_282, t_283);
not(t_284, u6);
not(t_285, x6);
or(p7, t_284, t_285);
not(t_286, z6);
not(t_287, a7);
or(q7, t_286, t_287);
not(t_288, b7);
not(t_289, v5);
or(r7, t_288, t_289);
not(t_290, c7);
not(t_291, w5);
or(s7, t_290, t_291);
not(t_292, d7);
not(t_293, x5);
or(t7, t_292, t_293);
not(t_294, e7);
not(t_295, y5);
or(u7, t_294, t_295);
not(t_296, c7);
not(t_297, z5);
or(v7, t_296, t_297);
not(t_298, f7);
not(t_299, a6);
or(w7, t_298, t_299);
not(t_300, g7);
not(t_301, b6);
or(x7, t_300, t_301);
not(t_302, g7);
not(t_303, c6);
or(y7, t_302, t_303);
not(t_304, b7);
not(t_305, d6);
or(z7, t_304, t_305);
not(t_306, h7);
not(t_307, e6);
or(a8, t_306, t_307);
not(t_308, d7);
not(t_309, f6);
or(b8, t_308, t_309);
not(t_310, e7);
not(t_311, g6);
or(c8, t_310, t_311);
not(t_312, h7);
not(t_313, h6);
or(d8, t_312, t_313);
not(t_314, f7);
not(t_315, i6);
or(e8, t_314, t_315);
not(t_316, i7);
not(t_317, j6);
or(f8, t_316, t_317);
not(t_318, i7);
not(t_319, k6);
or(g8, t_318, t_319);
not(t_320, j7);
not(t_321, l6);
or(h8, t_320, t_321);
not(t_322, k7);
not(t_323, m6);
or(i8, t_322, t_323);
not(t_324, l7);
not(t_325, n6);
or(j8, t_324, t_325);
not(t_326, m7);
not(t_327, o6);
or(k8, t_326, t_327);
not(t_328, k7);
not(t_329, p6);
or(l8, t_328, t_329);
not(t_330, n7);
not(t_331, q6);
or(m8, t_330, t_331);
not(t_332, o7);
not(t_333, r6);
or(n8, t_332, t_333);
not(t_334, o7);
not(t_335, s6);
or(o8, t_334, t_335);
not(t_336, j7);
not(t_337, t6);
or(p8, t_336, t_337);
not(t_338, p7);
not(t_339, u6);
or(q8, t_338, t_339);
not(t_340, l7);
not(t_341, v6);
or(r8, t_340, t_341);
not(t_342, m7);
not(t_343, w6);
or(s8, t_342, t_343);
not(t_344, p7);
not(t_345, x6);
or(t8, t_344, t_345);
not(t_346, n7);
not(t_347, y6);
or(u8, t_346, t_347);
not(t_348, q7);
not(t_349, z6);
or(v8, t_348, t_349);
not(t_350, q7);
not(t_351, a7);
or(w8, t_350, t_351);
not(t_352, r7);
not(t_353, z7);
or(x8, t_352, t_353);
not(t_354, s7);
not(t_355, v7);
or(y8, t_354, t_355);
not(t_356, t7);
not(t_357, b8);
or(z8, t_356, t_357);
not(t_358, u7);
not(t_359, c8);
or(a9, t_358, t_359);
not(t_360, w7);
not(t_361, e8);
or(b9, t_360, t_361);
not(t_362, x7);
not(t_363, y7);
or(c9, t_362, t_363);
not(t_364, a8);
not(t_365, d8);
or(d9, t_364, t_365);
not(t_366, f8);
not(t_367, g8);
or(e9, t_366, t_367);
not(t_368, h8);
not(t_369, p8);
or(f9, t_368, t_369);
not(t_370, i8);
not(t_371, l8);
or(g9, t_370, t_371);
not(t_372, j8);
not(t_373, r8);
or(h9, t_372, t_373);
not(t_374, k8);
not(t_375, s8);
or(i9, t_374, t_375);
not(t_376, m8);
not(t_377, u8);
or(j9, t_376, t_377);
not(t_378, n8);
not(t_379, o8);
or(k9, t_378, t_379);
not(t_380, q8);
not(t_381, t8);
or(l9, t_380, t_381);
not(t_382, v8);
not(t_383, w8);
or(m9, t_382, t_383);
not(t_384, y8);
not(t_385, c9);
or(n9, t_384, t_385);
not(t_386, y8);
not(t_387, d9);
or(o9, t_386, t_387);
not(t_388, c9);
not(t_389, e9);
or(p9, t_388, t_389);
not(t_390, d9);
not(t_391, e9);
or(q9, t_390, t_391);
not(t_392, g9);
not(t_393, k9);
or(r9, t_392, t_393);
not(t_394, g9);
not(t_395, l9);
or(s9, t_394, t_395);
not(t_396, k9);
not(t_397, m9);
or(t9, t_396, t_397);
not(t_398, l9);
not(t_399, m9);
or(u9, t_398, t_399);
not(t_400, n9);
not(t_401, y8);
or(v9, t_400, t_401);
not(t_402, o9);
not(t_403, y8);
or(w9, t_402, t_403);
not(t_404, n9);
not(t_405, c9);
or(x9, t_404, t_405);
not(t_406, p9);
not(t_407, c9);
or(y9, t_406, t_407);
not(t_408, q9);
not(t_409, d9);
or(z9, t_408, t_409);
not(t_410, o9);
not(t_411, d9);
or(a10, t_410, t_411);
not(t_412, q9);
not(t_413, e9);
or(b10, t_412, t_413);
not(t_414, p9);
not(t_415, e9);
or(c10, t_414, t_415);
not(t_416, r9);
not(t_417, g9);
or(d10, t_416, t_417);
not(t_418, s9);
not(t_419, g9);
or(e10, t_418, t_419);
not(t_420, r9);
not(t_421, k9);
or(f10, t_420, t_421);
not(t_422, t9);
not(t_423, k9);
or(g10, t_422, t_423);
not(t_424, u9);
not(t_425, l9);
or(h10, t_424, t_425);
not(t_426, s9);
not(t_427, l9);
or(i10, t_426, t_427);
not(t_428, u9);
not(t_429, m9);
or(j10, t_428, t_429);
not(t_430, t9);
not(t_431, m9);
or(k10, t_430, t_431);
not(t_432, v9);
not(t_433, x9);
or(l10, t_432, t_433);
not(t_434, w9);
not(t_435, a10);
or(m10, t_434, t_435);
not(t_436, y9);
not(t_437, c10);
or(n10, t_436, t_437);
not(t_438, z9);
not(t_439, b10);
or(o10, t_438, t_439);
not(t_440, d10);
not(t_441, f10);
or(p10, t_440, t_441);
not(t_442, e10);
not(t_443, i10);
or(q10, t_442, t_443);
not(t_444, g10);
not(t_445, k10);
or(r10, t_444, t_445);
not(t_446, h10);
not(t_447, j10);
or(s10, t_446, t_447);
not(t_448, q10);
not(t_449, v1);
or(t10, t_448, t_449);
not(t_450, r10);
not(t_451, w1);
or(u10, t_450, t_451);
not(t_452, p10);
not(t_453, x1);
or(v10, t_452, t_453);
not(t_454, s10);
not(t_455, y1);
or(w10, t_454, t_455);
not(t_456, m10);
not(t_457, z1);
or(x10, t_456, t_457);
not(t_458, n10);
not(t_459, a2);
or(y10, t_458, t_459);
not(t_460, l10);
not(t_461, b2);
or(z10, t_460, t_461);
not(t_462, o10);
not(t_463, c2);
or(a11, t_462, t_463);
not(t_464, t10);
not(t_465, v1);
or(b11, t_464, t_465);
not(t_466, u10);
not(t_467, w1);
or(c11, t_466, t_467);
not(t_468, v10);
not(t_469, x1);
or(d11, t_468, t_469);
not(t_470, w10);
not(t_471, y1);
or(e11, t_470, t_471);
not(t_472, x10);
not(t_473, z1);
or(f11, t_472, t_473);
not(t_474, y10);
not(t_475, a2);
or(g11, t_474, t_475);
not(t_476, z10);
not(t_477, b2);
or(h11, t_476, t_477);
not(t_478, a11);
not(t_479, c2);
or(i11, t_478, t_479);
not(t_480, z10);
not(t_481, l10);
or(j11, t_480, t_481);
not(t_482, x10);
not(t_483, m10);
or(k11, t_482, t_483);
not(t_484, y10);
not(t_485, n10);
or(l11, t_484, t_485);
not(t_486, a11);
not(t_487, o10);
or(m11, t_486, t_487);
not(t_488, v10);
not(t_489, p10);
or(n11, t_488, t_489);
not(t_490, t10);
not(t_491, q10);
or(o11, t_490, t_491);
not(t_492, u10);
not(t_493, r10);
or(p11, t_492, t_493);
not(t_494, w10);
not(t_495, s10);
or(q11, t_494, t_495);
not(t_496, o11);
not(t_497, b11);
or(r11, t_496, t_497);
not(t_498, p11);
not(t_499, c11);
or(s11, t_498, t_499);
not(t_500, n11);
not(t_501, d11);
or(t11, t_500, t_501);
not(t_502, q11);
not(t_503, e11);
or(u11, t_502, t_503);
not(t_504, k11);
not(t_505, f11);
or(v11, t_504, t_505);
not(t_506, l11);
not(t_507, g11);
or(w11, t_506, t_507);
not(t_508, j11);
not(t_509, h11);
or(x11, t_508, t_509);
not(t_510, m11);
not(t_511, i11);
or(y11, t_510, t_511);
not(t_512, r11);
not(t_513, x8);
or(z11, t_512, t_513);
not(t_514, s11);
not(t_515, z8);
or(a12, t_514, t_515);
not(t_516, t11);
not(t_517, a9);
or(b12, t_516, t_517);
not(t_518, u11);
not(t_519, b9);
or(c12, t_518, t_519);
not(t_520, v11);
not(t_521, f9);
or(d12, t_520, t_521);
not(t_522, w11);
not(t_523, h9);
or(e12, t_522, t_523);
not(t_524, x11);
not(t_525, i9);
or(f12, t_524, t_525);
not(t_526, y11);
not(t_527, j9);
or(g12, t_526, t_527);
not(t_528, z11);
not(t_529, r11);
or(h12, t_528, t_529);
not(t_530, a12);
not(t_531, s11);
or(i12, t_530, t_531);
not(t_532, b12);
not(t_533, t11);
or(j12, t_532, t_533);
not(t_534, c12);
not(t_535, u11);
or(k12, t_534, t_535);
not(t_536, d12);
not(t_537, v11);
or(l12, t_536, t_537);
not(t_538, e12);
not(t_539, w11);
or(m12, t_538, t_539);
not(t_540, f12);
not(t_541, x11);
or(n12, t_540, t_541);
not(t_542, g12);
not(t_543, y11);
or(o12, t_542, t_543);
not(t_544, z11);
not(t_545, x8);
or(p12, t_544, t_545);
not(t_546, a12);
not(t_547, z8);
or(q12, t_546, t_547);
not(t_548, b12);
not(t_549, a9);
or(r12, t_548, t_549);
not(t_550, c12);
not(t_551, b9);
or(s12, t_550, t_551);
not(t_552, d12);
not(t_553, f9);
or(t12, t_552, t_553);
not(t_554, e12);
not(t_555, h9);
or(u12, t_554, t_555);
not(t_556, f12);
not(t_557, i9);
or(v12, t_556, t_557);
not(t_558, g12);
not(t_559, j9);
or(w12, t_558, t_559);
not(t_560, h12);
not(t_561, p12);
or(x12, t_560, t_561);
not(t_562, i12);
not(t_563, q12);
or(y12, t_562, t_563);
not(t_564, j12);
not(t_565, r12);
or(z12, t_564, t_565);
not(t_566, k12);
not(t_567, s12);
or(a13, t_566, t_567);
not(t_568, l12);
not(t_569, t12);
or(b13, t_568, t_569);
not(t_570, m12);
not(t_571, u12);
or(c13, t_570, t_571);
not(t_572, n12);
not(t_573, v12);
or(d13, t_572, t_573);
not(t_574, o12);
not(t_575, w12);
or(e13, t_574, t_575);
not(f13, x12);
not(g13, x12);
not(h13, x12);
not(i13, x12);
not(j13, x12);
not(k13, y12);
not(l13, y12);
not(m13, y12);
not(n13, y12);
not(o13, y12);
not(p13, z12);
not(q13, z12);
not(r13, z12);
not(s13, z12);
not(t13, z12);
not(u13, a13);
not(v13, a13);
not(w13, a13);
not(x13, a13);
not(y13, a13);
not(z13, b13);
not(a14, b13);
not(b14, b13);
not(c14, b13);
not(d14, b13);
not(e14, c13);
not(f14, c13);
not(g14, c13);
not(h14, c13);
not(i14, c13);
not(j14, d13);
not(k14, d13);
not(l14, d13);
not(m14, d13);
not(n14, d13);
not(o14, e13);
not(p14, e13);
not(q14, e13);
not(r14, e13);
not(s14, e13);
and(t14, w13, r13, m13, x12);
and(u14, x13, s13, y12, h13);
and(v14, y13, z12, n13, i13);
and(w14, a13, t13, o13, j13);
and(x14, o14, j14, e14, b13);
and(y14, p14, k14, c13, z13);
and(z14, q14, d13, f14, a14);
and(a15, e13, l14, g14, b14);
or(b15, w14, v14, u14, t14);
or(c15, a15, z14, y14, x14);
and(d15, a13, q13, k13, x12, c15);
and(e15, v13, z12, l13, x12, c15);
and(f15, a13, p13, y12, f13, c15);
and(g15, u13, z12, y12, g13, c15);
and(h15, e13, n14, h14, b13, b15);
and(i15, s14, d13, i14, b13, b15);
and(j15, e13, m14, c13, c14, b15);
and(k15, r14, d13, c13, d14, b15);
and(l15, x12, j15);
and(m15, x12, h15);
and(n15, x12, k15);
and(o15, x12, i15);
and(p15, y12, j15);
and(q15, y12, h15);
and(r15, y12, k15);
and(s15, y12, i15);
and(t15, z12, j15);
and(u15, z12, h15);
and(v15, z12, k15);
and(w15, z12, i15);
and(x15, a13, j15);
and(y15, a13, h15);
and(z15, a13, k15);
and(a16, a13, i15);
and(b16, b13, f15);
and(c16, b13, d15);
and(d16, b13, g15);
and(e16, b13, e15);
and(f16, c13, f15);
and(g16, c13, d15);
and(h16, c13, g15);
and(i16, c13, e15);
and(j16, d13, f15);
and(k16, d13, d15);
and(l16, d13, g15);
and(m16, d13, e15);
and(n16, e13, f15);
and(o16, e13, d15);
and(p16, e13, g15);
and(q16, e13, e15);
not(t_576, o15);
not(t_577, f0);
or(r16, t_576, t_577);
not(t_578, s15);
not(t_579, e0);
or(s16, t_578, t_579);
not(t_580, w15);
not(t_581, d0);
or(t16, t_580, t_581);
not(t_582, a16);
not(t_583, c0);
or(u16, t_582, t_583);
not(t_584, n15);
not(t_585, b0);
or(v16, t_584, t_585);
not(t_586, r15);
not(t_587, a0);
or(w16, t_586, t_587);
not(t_588, v15);
not(t_589, z);
or(x16, t_588, t_589);
not(t_590, z15);
not(t_591, y);
or(y16, t_590, t_591);
not(t_592, m15);
not(t_593, x);
or(z16, t_592, t_593);
not(t_594, q15);
not(t_595, w);
or(a17, t_594, t_595);
not(t_596, u15);
not(t_597, v);
or(b17, t_596, t_597);
not(t_598, y15);
not(t_599, u);
or(c17, t_598, t_599);
not(t_600, l15);
not(t_601, t);
or(d17, t_600, t_601);
not(t_602, p15);
not(t_603, s);
or(e17, t_602, t_603);
not(t_604, t15);
not(t_605, r);
or(f17, t_604, t_605);
not(t_606, x15);
not(t_607, q);
or(g17, t_606, t_607);
not(t_608, e16);
not(t_609, p);
or(h17, t_608, t_609);
not(t_610, i16);
not(t_611, o);
or(i17, t_610, t_611);
not(t_612, m16);
not(t_613, n);
or(j17, t_612, t_613);
not(t_614, q16);
not(t_615, m);
or(k17, t_614, t_615);
not(t_616, d16);
not(t_617, l);
or(l17, t_616, t_617);
not(t_618, h16);
not(t_619, k);
or(m17, t_618, t_619);
not(t_620, l16);
not(t_621, j);
or(n17, t_620, t_621);
not(t_622, p16);
not(t_623, i);
or(o17, t_622, t_623);
not(t_624, c16);
not(t_625, h);
or(p17, t_624, t_625);
not(t_626, g16);
not(t_627, g);
or(q17, t_626, t_627);
not(t_628, k16);
not(t_629, f);
or(r17, t_628, t_629);
not(t_630, o16);
not(t_631, e);
or(s17, t_630, t_631);
not(t_632, b16);
not(t_633, d);
or(t17, t_632, t_633);
not(t_634, f16);
not(t_635, c);
or(u17, t_634, t_635);
not(t_636, j16);
not(t_637, b);
or(v17, t_636, t_637);
not(t_638, n16);
not(t_639, a);
or(w17, t_638, t_639);
not(t_640, d17);
not(t_641, l15);
or(x17, t_640, t_641);
not(t_642, z16);
not(t_643, m15);
or(y17, t_642, t_643);
not(t_644, v16);
not(t_645, n15);
or(z17, t_644, t_645);
not(t_646, r16);
not(t_647, o15);
or(a18, t_646, t_647);
not(t_648, e17);
not(t_649, p15);
or(b18, t_648, t_649);
not(t_650, a17);
not(t_651, q15);
or(c18, t_650, t_651);
not(t_652, w16);
not(t_653, r15);
or(d18, t_652, t_653);
not(t_654, s16);
not(t_655, s15);
or(e18, t_654, t_655);
not(t_656, f17);
not(t_657, t15);
or(f18, t_656, t_657);
not(t_658, b17);
not(t_659, u15);
or(g18, t_658, t_659);
not(t_660, x16);
not(t_661, v15);
or(h18, t_660, t_661);
not(t_662, t16);
not(t_663, w15);
or(i18, t_662, t_663);
not(t_664, g17);
not(t_665, x15);
or(j18, t_664, t_665);
not(t_666, c17);
not(t_667, y15);
or(k18, t_666, t_667);
not(t_668, y16);
not(t_669, z15);
or(l18, t_668, t_669);
not(t_670, u16);
not(t_671, a16);
or(m18, t_670, t_671);
not(t_672, t17);
not(t_673, b16);
or(n18, t_672, t_673);
not(t_674, p17);
not(t_675, c16);
or(o18, t_674, t_675);
not(t_676, l17);
not(t_677, d16);
or(p18, t_676, t_677);
not(t_678, h17);
not(t_679, e16);
or(q18, t_678, t_679);
not(t_680, u17);
not(t_681, f16);
or(r18, t_680, t_681);
not(t_682, q17);
not(t_683, g16);
or(s18, t_682, t_683);
not(t_684, m17);
not(t_685, h16);
or(t18, t_684, t_685);
not(t_686, i17);
not(t_687, i16);
or(u18, t_686, t_687);
not(t_688, v17);
not(t_689, j16);
or(v18, t_688, t_689);
not(t_690, r17);
not(t_691, k16);
or(w18, t_690, t_691);
not(t_692, n17);
not(t_693, l16);
or(x18, t_692, t_693);
not(t_694, j17);
not(t_695, m16);
or(y18, t_694, t_695);
not(t_696, w17);
not(t_697, n16);
or(z18, t_696, t_697);
not(t_698, s17);
not(t_699, o16);
or(a19, t_698, t_699);
not(t_700, o17);
not(t_701, p16);
or(b19, t_700, t_701);
not(t_702, k17);
not(t_703, q16);
or(c19, t_702, t_703);
not(t_704, r16);
not(t_705, f0);
or(d19, t_704, t_705);
not(t_706, s16);
not(t_707, e0);
or(e19, t_706, t_707);
not(t_708, t16);
not(t_709, d0);
or(f19, t_708, t_709);
not(t_710, u16);
not(t_711, c0);
or(g19, t_710, t_711);
not(t_712, v16);
not(t_713, b0);
or(h19, t_712, t_713);
not(t_714, w16);
not(t_715, a0);
or(i19, t_714, t_715);
not(t_716, x16);
not(t_717, z);
or(j19, t_716, t_717);
not(t_718, y16);
not(t_719, y);
or(k19, t_718, t_719);
not(t_720, z16);
not(t_721, x);
or(l19, t_720, t_721);
not(t_722, a17);
not(t_723, w);
or(m19, t_722, t_723);
not(t_724, b17);
not(t_725, v);
or(n19, t_724, t_725);
not(t_726, c17);
not(t_727, u);
or(o19, t_726, t_727);
not(t_728, d17);
not(t_729, t);
or(p19, t_728, t_729);
not(t_730, e17);
not(t_731, s);
or(q19, t_730, t_731);
not(t_732, f17);
not(t_733, r);
or(r19, t_732, t_733);
not(t_734, g17);
not(t_735, q);
or(s19, t_734, t_735);
not(t_736, h17);
not(t_737, p);
or(t19, t_736, t_737);
not(t_738, i17);
not(t_739, o);
or(u19, t_738, t_739);
not(t_740, j17);
not(t_741, n);
or(v19, t_740, t_741);
not(t_742, k17);
not(t_743, m);
or(w19, t_742, t_743);
not(t_744, l17);
not(t_745, l);
or(x19, t_744, t_745);
not(t_746, m17);
not(t_747, k);
or(y19, t_746, t_747);
not(t_748, n17);
not(t_749, j);
or(z19, t_748, t_749);
not(t_750, o17);
not(t_751, i);
or(a20, t_750, t_751);
not(t_752, p17);
not(t_753, h);
or(b20, t_752, t_753);
not(t_754, q17);
not(t_755, g);
or(c20, t_754, t_755);
not(t_756, r17);
not(t_757, f);
or(d20, t_756, t_757);
not(t_758, s17);
not(t_759, e);
or(e20, t_758, t_759);
not(t_760, t17);
not(t_761, d);
or(f20, t_760, t_761);
not(t_762, u17);
not(t_763, c);
or(g20, t_762, t_763);
not(t_764, v17);
not(t_765, b);
or(h20, t_764, t_765);
not(t_766, w17);
not(t_767, a);
or(i20, t_766, t_767);
not(t_768, x17);
not(t_769, p19);
or(i1, t_768, t_769);
not(t_770, y17);
not(t_771, l19);
or(m1, t_770, t_771);
not(t_772, z17);
not(t_773, h19);
or(q1, t_772, t_773);
not(t_774, a18);
not(t_775, d19);
or(u1, t_774, t_775);
not(t_776, b18);
not(t_777, q19);
or(h1, t_776, t_777);
not(t_778, c18);
not(t_779, m19);
or(l1, t_778, t_779);
not(t_780, d18);
not(t_781, i19);
or(p1, t_780, t_781);
not(t_782, e18);
not(t_783, e19);
or(t1, t_782, t_783);
not(t_784, f18);
not(t_785, r19);
or(g1, t_784, t_785);
not(t_786, g18);
not(t_787, n19);
or(k1, t_786, t_787);
not(t_788, h18);
not(t_789, j19);
or(o1, t_788, t_789);
not(t_790, i18);
not(t_791, f19);
or(s1, t_790, t_791);
not(t_792, j18);
not(t_793, s19);
or(f1, t_792, t_793);
not(t_794, k18);
not(t_795, o19);
or(j1, t_794, t_795);
not(t_796, l18);
not(t_797, k19);
or(n1, t_796, t_797);
not(t_798, m18);
not(t_799, g19);
or(r1, t_798, t_799);
not(t_800, n18);
not(t_801, f20);
or(s0, t_800, t_801);
not(t_802, o18);
not(t_803, b20);
or(w0, t_802, t_803);
not(t_804, p18);
not(t_805, x19);
or(a1, t_804, t_805);
not(t_806, q18);
not(t_807, t19);
or(e1, t_806, t_807);
not(t_808, r18);
not(t_809, g20);
or(r0, t_808, t_809);
not(t_810, s18);
not(t_811, c20);
or(v0, t_810, t_811);
not(t_812, t18);
not(t_813, y19);
or(z0, t_812, t_813);
not(t_814, u18);
not(t_815, u19);
or(d1, t_814, t_815);
not(t_816, v18);
not(t_817, h20);
or(q0, t_816, t_817);
not(t_818, w18);
not(t_819, d20);
or(u0, t_818, t_819);
not(t_820, x18);
not(t_821, z19);
or(y0, t_820, t_821);
not(t_822, y18);
not(t_823, v19);
or(c1, t_822, t_823);
not(t_824, z18);
not(t_825, i20);
or(p0, t_824, t_825);
not(t_826, a19);
not(t_827, e20);
or(t0, t_826, t_827);
not(t_828, b19);
not(t_829, a20);
or(x0, t_828, t_829);
not(t_830, c19);
not(t_831, w19);
or(b1, t_830, t_831);
endmodule
module top;
	parameter in_width = 41,
		patterns = 5000,
		step = 1;
	reg [1:in_width] in_mem[1:patterns];
	integer index;

	wire i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,
		i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,
		i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,
		i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,
		i40;

	assign {i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,
		i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,
		i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,
		i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,
		i40} = 
		$getpattern(in_mem[index]);

	initial $monitor($time,,o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,
		o10,o11,o12,o13,o14,o15,o16,o17,o18,o19,
		o20,o21,o22,o23,o24,o25,o26,o27,o28,o29,
		o30,o31);
	initial
		begin
			$readmemb("patt.mem", in_mem);
			for(index = 1; index <= patterns; index = index + 1)
				#step;
		end

	foobar cct(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,
		o10,o11,o12,o13,o14,o15,o16,o17,o18,o19,
		o20,o21,o22,o23,o24,o25,o26,o27,o28,o29,
		o30,o31,i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,
		i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,
		i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,
		i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,
		i40);
endmodule
