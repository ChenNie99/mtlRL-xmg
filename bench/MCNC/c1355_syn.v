/* Generated by Yosys 0.9 (git sha1 1979e0b) */

(* top =  1  *)
(* src = "c1355.v:1" *)
module c1355(G1, G10, G11, G12, G13, G1324, G1325, G1326, G1327, G1328, G1329, G1330, G1331, G1332, G1333, G1334, G1335, G1336, G1337, G1338, G1339, G1340, G1341, G1342, G1343, G1344, G1345, G1346, G1347, G1348, G1349, G1350, G1351, G1352, G1353, G1354, G1355, G14, G15, G16, G17, G18, G19, G2, G20, G21, G22, G23, G24, G25, G26, G27, G28, G29, G3, G30, G31, G32, G33, G34, G35, G36, G37, G38, G39, G4, G40, G41, G5, G6, G7, G8, G9);
  wire _000_;
  wire _001_;
  wire _002_;
  wire _003_;
  wire _004_;
  wire _005_;
  wire _006_;
  wire _007_;
  wire _008_;
  wire _009_;
  wire _010_;
  wire _011_;
  wire _012_;
  wire _013_;
  wire _014_;
  wire _015_;
  wire _016_;
  wire _017_;
  wire _018_;
  wire _019_;
  wire _020_;
  wire _021_;
  wire _022_;
  wire _023_;
  wire _024_;
  wire _025_;
  wire _026_;
  wire _027_;
  wire _028_;
  wire _029_;
  wire _030_;
  wire _031_;
  wire _032_;
  wire _033_;
  wire _034_;
  wire _035_;
  wire _036_;
  wire _037_;
  wire _038_;
  wire _039_;
  wire _040_;
  wire _041_;
  wire _042_;
  wire _043_;
  wire _044_;
  wire _045_;
  wire _046_;
  wire _047_;
  wire _048_;
  wire _049_;
  wire _050_;
  wire _051_;
  wire _052_;
  wire _053_;
  wire _054_;
  wire _055_;
  wire _056_;
  wire _057_;
  wire _058_;
  wire _059_;
  wire _060_;
  wire _061_;
  wire _062_;
  wire _063_;
  wire _064_;
  wire _065_;
  wire _066_;
  wire _067_;
  wire _068_;
  wire _069_;
  wire _070_;
  wire _071_;
  wire _072_;
  wire _073_;
  wire _074_;
  wire _075_;
  wire _076_;
  wire _077_;
  wire _078_;
  wire _079_;
  wire _080_;
  wire _081_;
  wire _082_;
  wire _083_;
  wire _084_;
  wire _085_;
  wire _086_;
  wire _087_;
  wire _088_;
  wire _089_;
  wire _090_;
  wire _091_;
  wire _092_;
  wire _093_;
  wire _094_;
  wire _095_;
  wire _096_;
  wire _097_;
  wire _098_;
  wire _099_;
  wire _100_;
  wire _101_;
  wire _102_;
  wire _103_;
  wire _104_;
  wire _105_;
  wire _106_;
  wire _107_;
  wire _108_;
  wire _109_;
  wire _110_;
  wire _111_;
  wire _112_;
  wire _113_;
  wire _114_;
  wire _115_;
  wire _116_;
  wire _117_;
  wire _118_;
  wire _119_;
  wire _120_;
  wire _121_;
  wire _122_;
  wire _123_;
  wire _124_;
  wire _125_;
  wire _126_;
  wire _127_;
  wire _128_;
  wire _129_;
  wire _130_;
  wire _131_;
  wire _132_;
  wire _133_;
  wire _134_;
  wire _135_;
  wire _136_;
  wire _137_;
  wire _138_;
  wire _139_;
  wire _140_;
  wire _141_;
  wire _142_;
  wire _143_;
  wire _144_;
  wire _145_;
  wire _146_;
  wire _147_;
  wire _148_;
  wire _149_;
  wire _150_;
  wire _151_;
  wire _152_;
  wire _153_;
  wire _154_;
  wire _155_;
  (* src = "c1355.v:6" *)
  input G1;
  (* src = "c1355.v:6" *)
  input G10;
  (* src = "c1355.v:6" *)
  input G11;
  (* src = "c1355.v:6" *)
  input G12;
  (* src = "c1355.v:6" *)
  input G13;
  (* src = "c1355.v:9" *)
  output G1324;
  (* src = "c1355.v:9" *)
  output G1325;
  (* src = "c1355.v:9" *)
  output G1326;
  (* src = "c1355.v:9" *)
  output G1327;
  (* src = "c1355.v:9" *)
  output G1328;
  (* src = "c1355.v:9" *)
  output G1329;
  (* src = "c1355.v:9" *)
  output G1330;
  (* src = "c1355.v:9" *)
  output G1331;
  (* src = "c1355.v:9" *)
  output G1332;
  (* src = "c1355.v:9" *)
  output G1333;
  (* src = "c1355.v:9" *)
  output G1334;
  (* src = "c1355.v:9" *)
  output G1335;
  (* src = "c1355.v:9" *)
  output G1336;
  (* src = "c1355.v:9" *)
  output G1337;
  (* src = "c1355.v:9" *)
  output G1338;
  (* src = "c1355.v:9" *)
  output G1339;
  (* src = "c1355.v:9" *)
  output G1340;
  (* src = "c1355.v:9" *)
  output G1341;
  (* src = "c1355.v:9" *)
  output G1342;
  (* src = "c1355.v:9" *)
  output G1343;
  (* src = "c1355.v:9" *)
  output G1344;
  (* src = "c1355.v:9" *)
  output G1345;
  (* src = "c1355.v:9" *)
  output G1346;
  (* src = "c1355.v:9" *)
  output G1347;
  (* src = "c1355.v:9" *)
  output G1348;
  (* src = "c1355.v:9" *)
  output G1349;
  (* src = "c1355.v:9" *)
  output G1350;
  (* src = "c1355.v:9" *)
  output G1351;
  (* src = "c1355.v:9" *)
  output G1352;
  (* src = "c1355.v:9" *)
  output G1353;
  (* src = "c1355.v:9" *)
  output G1354;
  (* src = "c1355.v:9" *)
  output G1355;
  (* src = "c1355.v:6" *)
  input G14;
  (* src = "c1355.v:6" *)
  input G15;
  (* src = "c1355.v:6" *)
  input G16;
  (* src = "c1355.v:6" *)
  input G17;
  (* src = "c1355.v:6" *)
  input G18;
  (* src = "c1355.v:6" *)
  input G19;
  (* src = "c1355.v:6" *)
  input G2;
  (* src = "c1355.v:6" *)
  input G20;
  (* src = "c1355.v:6" *)
  input G21;
  (* src = "c1355.v:6" *)
  input G22;
  (* src = "c1355.v:6" *)
  input G23;
  (* src = "c1355.v:6" *)
  input G24;
  (* src = "c1355.v:6" *)
  input G25;
  (* src = "c1355.v:6" *)
  input G26;
  (* src = "c1355.v:6" *)
  input G27;
  (* src = "c1355.v:6" *)
  input G28;
  (* src = "c1355.v:6" *)
  input G29;
  (* src = "c1355.v:6" *)
  input G3;
  (* src = "c1355.v:6" *)
  input G30;
  (* src = "c1355.v:6" *)
  input G31;
  (* src = "c1355.v:6" *)
  input G32;
  (* src = "c1355.v:6" *)
  input G33;
  (* src = "c1355.v:6" *)
  input G34;
  (* src = "c1355.v:6" *)
  input G35;
  (* src = "c1355.v:6" *)
  input G36;
  (* src = "c1355.v:6" *)
  input G37;
  (* src = "c1355.v:6" *)
  input G38;
  (* src = "c1355.v:6" *)
  input G39;
  (* src = "c1355.v:6" *)
  input G4;
  (* src = "c1355.v:6" *)
  input G40;
  (* src = "c1355.v:6" *)
  input G41;
  (* src = "c1355.v:6" *)
  input G5;
  (* src = "c1355.v:6" *)
  input G6;
  (* src = "c1355.v:6" *)
  input G7;
  (* src = "c1355.v:6" *)
  input G8;
  (* src = "c1355.v:6" *)
  input G9;
  assign _130_ = G5 ^ G1;
  assign _131_ = ~(G13 ^ G9);
  assign _132_ = ~(_131_ ^ _130_);
  assign _133_ = G41 & G33;
  assign _134_ = ~(G18 ^ G17);
  assign _135_ = ~(G20 ^ G19);
  assign _136_ = _135_ ^ _134_;
  assign _137_ = G22 ^ G21;
  assign _138_ = ~(G24 ^ G23);
  assign _139_ = ~(_138_ ^ _137_);
  assign _140_ = ~(_139_ ^ _136_);
  assign _141_ = _140_ ^ _133_;
  assign _142_ = _141_ ^ _132_;
  assign _143_ = G21 ^ G17;
  assign _144_ = ~(G29 ^ G25);
  assign _145_ = ~(_144_ ^ _143_);
  assign _146_ = G37 & G41;
  assign _147_ = ~(G2 ^ G1);
  assign _148_ = ~(G4 ^ G3);
  assign _149_ = _148_ ^ _147_;
  assign _150_ = G6 ^ G5;
  assign _151_ = ~(G8 ^ G7);
  assign _152_ = ~(_151_ ^ _150_);
  assign _153_ = ~(_152_ ^ _149_);
  assign _154_ = _153_ ^ _146_;
  assign _155_ = ~(_154_ ^ _145_);
  assign _000_ = G22 ^ G18;
  assign _001_ = ~(G30 ^ G26);
  assign _002_ = ~(_001_ ^ _000_);
  assign _003_ = G38 & G41;
  assign _004_ = G10 ^ G9;
  assign _005_ = ~(G12 ^ G11);
  assign _006_ = _005_ ^ _004_;
  assign _007_ = G14 ^ G13;
  assign _008_ = ~(G16 ^ G15);
  assign _009_ = _008_ ^ _007_;
  assign _010_ = ~(_009_ ^ _006_);
  assign _011_ = _010_ ^ _003_;
  assign _012_ = _011_ ^ _002_;
  assign _013_ = ~(_012_ & _155_);
  assign _014_ = G23 ^ G19;
  assign _015_ = ~(G31 ^ G27);
  assign _016_ = ~(_015_ ^ _014_);
  assign _017_ = G39 & G41;
  assign _018_ = _006_ ^ _149_;
  assign _019_ = _018_ ^ _017_;
  assign _020_ = ~(_019_ ^ _016_);
  assign _021_ = _020_ & ~(_013_);
  assign _022_ = G24 ^ G20;
  assign _023_ = ~(G32 ^ G28);
  assign _024_ = _023_ ^ _022_;
  assign _025_ = G40 & G41;
  assign _026_ = _009_ ^ _152_;
  assign _027_ = _026_ ^ _025_;
  assign _028_ = ~(_027_ ^ _024_);
  assign _029_ = ~_028_;
  assign _030_ = _021_ & ~(_029_);
  assign _031_ = G8 ^ G4;
  assign _032_ = ~(G16 ^ G12);
  assign _033_ = _032_ ^ _031_;
  assign _034_ = G36 & G41;
  assign _035_ = G30 ^ G29;
  assign _036_ = ~(G32 ^ G31);
  assign _037_ = _036_ ^ _035_;
  assign _038_ = _037_ ^ _139_;
  assign _039_ = _038_ ^ _034_;
  assign _040_ = ~(_039_ ^ _033_);
  assign _041_ = ~_040_;
  assign _042_ = G6 ^ G2;
  assign _043_ = ~(G14 ^ G10);
  assign _044_ = ~(_043_ ^ _042_);
  assign _045_ = G34 & G41;
  assign _046_ = G26 ^ G25;
  assign _047_ = ~(G28 ^ G27);
  assign _048_ = _047_ ^ _046_;
  assign _049_ = ~(_037_ ^ _048_);
  assign _050_ = _049_ ^ _045_;
  assign _051_ = ~(_050_ ^ _044_);
  assign _052_ = _051_ | ~(_142_);
  assign _053_ = G7 ^ G3;
  assign _054_ = ~(G15 ^ G11);
  assign _055_ = ~(_054_ ^ _053_);
  assign _056_ = G35 & G41;
  assign _057_ = _048_ ^ _136_;
  assign _058_ = _057_ ^ _056_;
  assign _059_ = ~(_058_ ^ _055_);
  assign _060_ = _059_ | _052_;
  assign _061_ = _058_ ^ _055_;
  assign _062_ = _061_ | _052_;
  assign _063_ = _040_ ? _062_ : _060_;
  assign _064_ = ~(_051_ & _142_);
  assign _065_ = _064_ | _059_;
  assign _066_ = ~((_065_ | _041_) & _063_);
  assign _067_ = _051_ | _142_;
  assign _068_ = _067_ | _059_;
  assign _069_ = _040_ & ~(_068_);
  assign _070_ = ~((_069_ | _066_) & _030_);
  assign _071_ = _070_ | _142_;
  assign G1324 = _071_ ^ G1;
  assign _072_ = _070_ | ~(_051_);
  assign G1325 = _072_ ^ G2;
  assign _073_ = _070_ | _061_;
  assign G1326 = _073_ ^ G3;
  assign _074_ = _070_ | _040_;
  assign G1327 = _074_ ^ G4;
  assign _075_ = _020_ | _013_;
  assign _076_ = _029_ & ~(_075_);
  assign _077_ = ~((_069_ | _066_) & _076_);
  assign _078_ = _077_ | _142_;
  assign G1328 = _078_ ^ G5;
  assign _079_ = _077_ | ~(_051_);
  assign G1329 = _079_ ^ G6;
  assign _080_ = _077_ | _061_;
  assign G1330 = _080_ ^ G7;
  assign _081_ = _077_ | _040_;
  assign G1331 = _081_ ^ G8;
  assign _082_ = _019_ ^ _016_;
  assign _083_ = _012_ | _155_;
  assign _084_ = _083_ | _082_;
  assign _085_ = _028_ & ~(_084_);
  assign _086_ = ~((_069_ | _066_) & _085_);
  assign _087_ = _086_ | _142_;
  assign G1332 = _087_ ^ G9;
  assign _088_ = _086_ | ~(_051_);
  assign G1333 = _088_ ^ G10;
  assign _089_ = _086_ | _061_;
  assign G1334 = _089_ ^ G11;
  assign _090_ = _086_ | _040_;
  assign G1335 = _090_ ^ G12;
  assign _091_ = _083_ | _020_;
  assign _092_ = _029_ & ~(_091_);
  assign _093_ = ~((_069_ | _066_) & _092_);
  assign _094_ = _093_ | _142_;
  assign G1336 = _094_ ^ G13;
  assign _095_ = _093_ | ~(_051_);
  assign G1337 = _095_ ^ G14;
  assign _096_ = _093_ | _061_;
  assign G1338 = _096_ ^ G15;
  assign _097_ = _093_ | _040_;
  assign G1339 = _097_ ^ G16;
  assign _098_ = _067_ | _061_;
  assign _099_ = _040_ & ~(_098_);
  assign _100_ = _155_ | ~(_012_);
  assign _101_ = _100_ | _020_;
  assign _102_ = _100_ | _082_;
  assign _103_ = _028_ ? _102_ : _101_;
  assign _104_ = ~((_091_ | _029_) & _103_);
  assign _105_ = _028_ & ~(_075_);
  assign _106_ = ~((_105_ | _104_) & _099_);
  assign _107_ = _106_ | ~(_155_);
  assign G1340 = _107_ ^ G17;
  assign _108_ = _106_ | _012_;
  assign G1341 = _108_ ^ G18;
  assign _109_ = _106_ | _082_;
  assign G1342 = _109_ ^ G19;
  assign _110_ = _106_ | _028_;
  assign G1343 = _110_ ^ G20;
  assign _111_ = _041_ & ~(_068_);
  assign _112_ = ~((_105_ | _104_) & _111_);
  assign _113_ = _112_ | ~(_155_);
  assign G1344 = _113_ ^ G21;
  assign _114_ = _112_ | _012_;
  assign G1345 = _114_ ^ G22;
  assign _115_ = _112_ | _082_;
  assign G1346 = _115_ ^ G23;
  assign _116_ = _112_ | _028_;
  assign G1347 = _116_ ^ G24;
  assign _117_ = _064_ | _061_;
  assign _118_ = _040_ & ~(_117_);
  assign _119_ = ~((_105_ | _104_) & _118_);
  assign _120_ = _119_ | ~(_155_);
  assign G1348 = _120_ ^ G25;
  assign _121_ = _119_ | _012_;
  assign G1349 = _121_ ^ G26;
  assign _122_ = _119_ | _082_;
  assign G1350 = _122_ ^ G27;
  assign _123_ = _119_ | _028_;
  assign G1351 = _123_ ^ G28;
  assign _124_ = _041_ & ~(_065_);
  assign _125_ = ~((_105_ | _104_) & _124_);
  assign _126_ = _125_ | ~(_155_);
  assign G1352 = _126_ ^ G29;
  assign _127_ = _125_ | _012_;
  assign G1353 = _127_ ^ G30;
  assign _128_ = _125_ | _082_;
  assign G1354 = _128_ ^ G31;
  assign _129_ = _125_ | _028_;
  assign G1355 = _129_ ^ G32;
endmodule
