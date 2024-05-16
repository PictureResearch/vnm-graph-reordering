#!/bin/bash

# dataset filenames
filename=(
G67
Fashion_MNIST_norm_10NN
mnist_test_norm_10NN
cryg10000
kmnist_norm_10NN
bloweybq
Goodwin_030
cz10228
crack
hangGlider_3
sit100
har_10NN
shuttle_eddy
shuttle_eddy
c-42
vsp_p0291_seymourl_iiasa
bundle1
ted_A_unscaled
ted_B
ted_B_unscaled
ted_A
ted_AB
ted_AB_unscaled
FA
lhr10
lhr10c
PGPgiantcompo
c-44
nopoly
TSOPF_FS_b162_c1
pkustk02
msc10848
p2p-Gnutella04
rajat06
wing_nodal
igbt3
lhr11c
lhr11
bcsstk17
usps_norm_5NN
vsp_c-30_data_data
CurlCurl_0
k3plates
m3plates
c-43
fe_4elt2
bips98_1450
coupled
wiki-RfA
cage10
t2dah_a
t2dah
t2dah_e
Oregon-1
sinc15
fd15
nopss_11k
inlet
pesa
g7jac040
g7jac040sc
Oregon-2
bcsstk18
linverse
ex19
ca-HepPh
ncvxqp1
circuit_3
mycielskian14
vibrobox
sme3Da
stokes64s
stokes64
skirt
skirt
tuma2
c-45
big
mimo46x46_system
xingo_afonso_itaipu
ww_vref_6405
mimo28x28_system
bips07_1693
zeros_nopss_13k
mimo8x8_system
Reuters911
foldoc
bayer10
poisson3Da
garon2
lowThrust_4
cyl6
cbuckle
jan99jac040
jan99jac040sc
bayer02
pcrystk02
crystm02
crystk02
bcsstk29
appu
TSOPF_RS_b39_c7
vsp_befref_fxm_2_4_air02
airfoil_2d
lhr14
lhr14c
human_gene2
case9
TSOPF_FS_b9_c6
TSOPF_RS_b300_c1
epb1
g7jac050sc
Pres_Poisson
rajat07
c-46
bips07_1998
c-47
TSOPF_RS_b162_c3
OPF_3754
bcsstm25
bcsstk25
opt1
hangGlider_4
poli_large
barth5
barth5
powersim
hangGlider_5
Dubcova1
olafu
lowThrust_5
net50
delaunay_n14
fe_sphere
sinc18
fd18
ncvxqp9
pds10
ex11
astro-ph
cond-mat
ex3sta1
gupta3
ramage02
cti
pkustk07
bips07_2476
lowThrust_6
poli3
Si10H16
copter1
e40r0100
gyro_m
gyro
gyro_k
lowThrust_7
cvxqp3
bodyy4
lhr17c
lhr17
lowThrust_8
g7jac060sc
g7jac060
memplus
FEM_3D_thermal1
Goodwin_040
L-9
nd6k
crplat2
lowThrust_9
lowThrust_10
mark3jac040sc
mark3jac040
c-48
lowThrust_11
tandem_vtx
lowThrust_12
lowThrust_13
shermanACb
nmos3
bodyy5
TSOPF_RS_b678_c1
ford1
ca-AstroPh
fxm4_6
whitaker3_dual
pattern1
rajat08
bodyy6
ex35
raefsky4
Si5H12
LFAT5000
LF10000
Trefethen_20000b
qpband
Trefethen_20000
ACTIVSg10K
worms20_10NN
chipcool0
chipcool1
crack_dual
rail_20209
t3dl_e
t3dl
t3dl_a
TSOPF_RS_b162_c4
ns3Da
cz20468
bayer04
jan99jac060sc
jan99jac060
rajat27
tsyl201
descriptor_xingo6u
Ill_Stokes
xingo3012
waveguide3D
bips07_3078
c-49
raefsky3
tube2
tube1
biplane-9
std1_Jac2
std1_Jac3_db
std1_Jac3
std1_Jac2_db
vsp_msc10848_300sep_100in_1Kout
pkustk01
trdheim
pkustk08
human_gene1
c-50
cs4
rim
p2p-Gnutella25
windscreen
pli
li
Zd_Jac3
Zd_Jac2
Zd_Jac6_db
Zd_Jac6
Zd_Jac3_db
Zd_Jac2_db
as-22july06
tuma1
msc23052
bcsstk36
bcsstm36
net75
ca-CondMat
c-51
EAT_SR
EAT_RS
ABACUS_shell_ud
ABACUS_shell_hd
ABACUS_shell_ld
ABACUS_shell_md
af23560
g7jac080sc
g7jac080
c-52
stufe-10
de2010
aug3d
rajat09
mycielskian15
crystk03
pcrystk03
crystm03
sx-mathoverflow
hvdc1
dtoc
hi2010
ri2010
mult_dcop_02
mult_dcop_01
mult_dcop_03
epb2
bcsstm37
bcsstk37
TSOPF_RS_b2052_c1
smt
wang3
wang4
p2p-Gnutella24
HEP-th
mark3jac060sc
mark3jac060
jan99jac080
jan99jac080sc
brainpc2
TEM27623
2D_27628_bjtcai
cit-HepTh
HEP-th-new
bratu3d
TSOPF_FS_b39_c7
TSOPF_RS_b300_c2
bcsstk30
3D_28984_Tetra
aug2d
sme3Db
TSOPF_FS_b300
TSOPF_FS_b300_c1
light_in_tissue
g7jac100
g7jac100sc
thread
OPF_6000
net100
mixtank_new
spmsrtls
bloweybl
bloweya
aug2dc
rajat10
c-53
bcsstm35
bcsstk35
big_dual
wathen100
invextr1_new
TSOPF_FS_b162_c3
cond-mat-2003
as-caida
c-54
gupta1
vsp_barth5_1Ksep_50in_5Kout
helm3d01
lpl1
Goodwin_054
vt2010
rgg_n_2_15_s0
se
delaunay_n15
viscoplastic2
c-55
SiO
poli4
Zhao1
Zhao2
pkustk09
jan99jac100
jan99jac100sc
cit-HepPh
ship_001
lhr34c
lhr34
aug3dcqp
g7jac120sc
g7jac120
bcsstk31
TSOPF_RS_b678_c2
c-56
nd12k
onetone1
onetone2
pdb1HYS
wathen120
shock-9
pwt
pwt
pwt
mark3jac080sc
mark3jac080
p2p-Gnutella30
email-Enron
net125
pkustk05
rajat15
Chevron1
finance256
c-58
viscorocks
c-57
TSOPF_RS_b39_c19
TSOPF_RS_b2383_c1
TSOPF_RS_b2383
kim1
mario001
k49_norm_10NN
bbmat
cage11
vsp_south31_slptsk
rajat22
obstclae
jnlbrng1
torsion1
vsp_sctap1-2b_and_seymourl
case39
juba40k
bauru5727
chem_master1
cond-mat-2005
TSOPF_FS_b162_c4
minsurfo
windtunnel_evap3d
cz40948
av41092
c-59
jan99jac120sc
jan99jac120
g7jac140
g7jac140sc
c-62ghs
c-62
TSOPF_RS_b300_c3
sme3Dc
pkustk06
net150
c-61
c-60
OPF_10000
c-63
bcsstk32
fe_body
mouse_gene
vsp_model1_crew1_cr42_south31
ak2010
3dtube
mark3jac100sc
mark3jac100
bcsstk39
bcsstm39
rma10
mosfet2
vanbody
g7jac160sc
g7jac160
c-65
xenon1
nh2010
gridgena
mycielskian16
conf5_4-8x8-15
bfly
cca
ccc
conf5_4-8x8-05
conf5_4-8x8-10
conf5_4-8x8-20
conf6_0-8x8-20
conf6_0-8x8-30
conf6_0-8x8-80
stokes128
ckt11752_tr_0
ckt11752_dc_1
c-66
c-66b
cvxbqp1
ncvxbqp1
sparsine
rajat26
c-64
c-64b
ibm_matrix_2
3D_51448_3D
dawson5
ecl32
pct20stif
ct20stif
dictionary28
crankseg_1
g7jac180sc
g7jac180
struct3
2D_54019_highK
nasasrb
srb1
mark3jac120
mark3jac120sc
copter2
copter2
pkustk04
Goodwin_071
vsp_bump2_e18_aa01_model1_crew1
TSOPF_FS_b300_c2
bayer01
c-67b
c-67
loc-Brightkite
vsp_bcsstk30_500sep_10in_1Kout
g7jac200sc
g7jac200
Andrews
dixmaanl
t60k
a5esindl
blockqp1
TSOPF_RS_b39_c30
water_tank
GaAsH6
Ga3As3H12
wing
gupta2
venkat25
venkat50
venkat01
cant
ncvxqp5
p2p-Gnutella31
brack2
pkustk03
crankseg_2
mark3jac140
mark3jac140sc
c-68
Dubcova2
kron_g500-logn16
delaunay_n16
rgg_n_2_16_s0
qa8fm
qa8fk
mip1
gas_sensor
H2O
laminar_duct3D
c-69
ct2010
k1_san
Chebyshev4
bcircuit
c-70
enron
me2010
ACTIVSg70K
lhr71
lhr71c
cfd1
F2
nd24k
oilpan
fem_filter
RFdevice
finan512
pfinan512
ncvxqp3
nv1
soc-Epinions1
TSOPF_FS_b39_c19
shyy161
c-71
vsp_vibrobox_scagr7-2c_rlfddd
soc-sign-Slashdot081106
soc-Slashdot0811
fe_tooth
t3dh_e
t3dh
t3dh_a
rail_79841
a2nnsnsl
a0nsdsil
circuit_4
cont-201
pkustk10
apache1
soc-sign-Slashdot090216
shallow_water2
shallow_water1
soc-sign-Slashdot090221
soc-Slashdot0902
thermal1
Wordnet3
consph
c-72
TSOPF_FS_b300_c3
nv2010
epb3
onera_dual
poisson3Db
vsp_c-60_data_cti_cs4
wy2010
rajat20
rajat28
rajat25
ncvxqp7
pkustk11
LeGresley_87936
olesnik0
net4-1
sd2010
denormal
Chevron2
s4dkt3m2
s3dkt3m2
s3dkq4m2
boyd1
vfem
tandem_dual
rajat18
rajat16
rajat17
pkustk12
pkustk13
ifiss_mat
Si34H36
m_t1
mycielskian17
ASIC_100ks
ASIC_100k
fe_rotor
G_n_pin_pout
smallworld
preferentialAttachment
Goodwin_095
ford2
vsp_mod2_pgp2_slptsk
2cubes_sphere
thermomech_TK
thermomech_TC
matrix_9
hcircuit
filter3D
x104
lung2
rajat23
598a
Baumann
Ge99H100
Ge87H76
barrier2-4
barrier2-3
barrier2-2
barrier2-1
Ga10As10H30
luxembourg_osm
shipsec8
ut2010
barrier2-9
barrier2-10
barrier2-11
barrier2-12
torso2
torso1
dc1
dc2
trans4
dc3
trans5
imagesensor
TSOPF_FS_b39_c30
twotone
cop20k_A
ship_003
cfd2
internet
matrix-new_3
usroads-48
boneS01
usroads
cage12
kron_g500-logn17
delaunay_n17
rgg_n_2_17_s0
soc-sign-epinions
mt2010
Ga19As19H42
nd2010
wv2010
vsp_finan512_scagr7-2c_rlfddd
shipsec1
ch7-8-b5
bmw7st_1
fe_ocean
engine
144
md2010
Dubcova3
FEM_3D_thermal2
bmwcra_1
id2010
G2_circuit
pkustk14
TEM152078
para-4
gearbox
SiO2
power9
para-7
para-5
para-10
para-6
para-9
para-8
wave
xenon2
ma2010
sx-askubuntu
majorbasis
crashbasis
PR02R
ky2010
nm2010
mono_500Hz
c-73b
c-73
nj2010
scircuit
ms2010
Goodwin_127
transient
shipsec5
cont-300
TEM181302
ohne2
sc2010
d_pretok
Si41Ge41H72
ar2010
hvdc2
turon_m
caidaRouterLevel
ne2010
sx-superuser
wa2010
loc-Gowalla
mycielskian18
or2010
power197k
fullb
shar_te2-b3
m133-b3
co2010
fcondp2
thermomech_dK
thermomech_dM
la2010
ss1
mac_econ_fwd500
stomach
troll
m14b
ia2010
pwtk
hood
CO
radiation
halfb
HTC_336_9129
HTC_336_4438
CurlCurl_1
coAuthorsCiteseer
bmw3_2
ks2010
tn2010
Si87H76
patents_main
az2010
BenElechi1
al2010
wi2010
Lin
torso3
mn2010
offshore
amazon0302
# delaunay_n18
# kron_g500-logn18
# rgg_n_2_18_s0
# Raj1
# email-EuAll
# in2010
# Ga41As41H72
# citationCiteseer
# ok2010
# iChem_Jacobian
# Stanford
# web-Stanford
# va2010
# nc2010
# ga2010
# 3Dspectralwave2
# coAuthorsDBLP
# analytics
# ins2
# com-DBLP
# ASIC_320ks
# ASIC_320k
# Linux_call_graph
# cnr-2000
# web-NotreDame
# NotreDame_www
# dblp-2010
# mi2010
# com-Amazon
# mo2010
# F1
# c-big
# ny2010
# rajat24
# oh2010
# ML_Laplace
# Chevron3
# RM07R
# mario002
# darcy003
# helm2d03
# test1
# mycielskian19
# language
# marine1
# amazon0312
# amazon0601
# amazon0505
# rajat21
# nxp1
# msdoor
# CoupCons3D
# dielFilterV3clx
# pa2010
# coPapersCiteseer
# largebasis
# cage13
# auto
# il2010
# higgs-twitter
# kim2
# boyd2
# fl2010
# fem_hifreq_circuit
# af_3_k101
# af_2_k101
# af_1_k101
# af_0_k101
# af_5_k101
# af_4_k101
# inline_1
# af_shell6
# af_shell7
# af_shell8
# af_shell9
# af_shell2
# af_shell3
# af_shell4
# af_shell5
# af_shell1
# bundle_adj
# delaunay_n19
# kron_g500-logn19
# rgg_n_2_19_s0
# mc2depi
# parabolic_fem
# lp1
# coPapersDBLP
# gsm_106857
# dielFilterV2clx
# Fault_639
# rajat30
# rajat29
# pre2
# 3Dspectralwave
# ASIC_680ks
# ASIC_680k
# Stanford_Berkeley
# web-BerkStan
# ca2010
# Chevron4
# apache2
# tmt_sym
# amazon-2008
# PFlow_742
# mycielskian20
# CurlCurl_2
# flickr
# eu-2005
# tx2010
# boneS10
# web-Google
# tmt_unsym
# t2em
# Emilia_923
# Hardesty1
# audikw_1
# ldoor
# bone010
# ecology2
# ecology1
# webbase-1M
# NACA0015
# kron_g500-logn20
# debr
# delaunay_n20
# rgg_n_2_20_s0
# nlpkkt80
# vas_stokes_1M
# roadNet-PA
# dielFilterV3real
# com-Youtube
# hollywood-2009
# wiki-talk-temporal
# dielFilterV2real
# dgreen
# CurlCurl_3
# thermal2
# atmosmodj
# atmosmodd
# in-2004
# Serena
# roadNet-TX
# Geo_1438
# belgium_osm
# Hamrle3
# nv2
# StocF-1465
# Long_Coup_dt6
# Long_Coup_dt0
# atmosmodl
# atmosmodm
# Hook_1498
# ML_Geer
# cage14
# af_shell10
# Flan_1565
# G3_circuit
# Transport
# soc-Pokec
# wikipedia-20051105
# ss
# as-Skitter
# wiki-topcats
# roadNet-CA
# HV15R
# kkt_power
# rgg_n_2_21_s0
# delaunay_n21
# kron_g500-logn21
# packing-500x100x100-b050
# vas_stokes_2M
# Cube_Coup_dt6
# Cube_Coup_dt0
# netherlands_osm
# CurlCurl_4
# wiki-Talk
# sx-stackoverflow
# memchip
# Bump_2911
# wikipedia-20060925
# FullChip
# Freescale2
# com-Orkut
# wikipedia-20061104
# Freescale1
# M6
# circuit5M_dc
# nlpkkt120
# wikipedia-20070206
# 333SP
# patents
# cit-Patents
# AS365
# com-LiveJournal
# venturiLevel3
# Queen_4147
# NLR
# delaunay_n22
# rgg_n_2_22_s0
# vas_stokes_4M
# hugetrace-00000
# rajat31
# channel-500x100x100-b050
# soc-LiveJournal1
# cage15
# ljournal-2008
# circuit5M
# hugetric-00000
# hugetric-00010
# italy_osm
# adaptive
# hugetric-00020
# indochina-2004
# great-britain_osm
# nlpkkt160
# rgg_n_2_23_s0
# delaunay_n23
# wb-edu
# stokes
# germany_osm
# asia_osm
# hugetrace-00010
# road_central
# hugetrace-00020
# nlpkkt200
# delaunay_n24
# rgg_n_2_24_s0
# hugebubbles-00000
# uk-2002
# mawi_201512012345
# hugebubbles-00010
# hugebubbles-00020
# arabic-2005
# road_usa
# GAP-road
# nlpkkt240
# MOLIERE_2016
# mawi_201512020000
# uk-2005
# it-2004
# twitter7
# GAP-web
# sk-2005
# europe_osm
# kmer_V2a
# GAP-twitter
# com-Friendster
)

######### group name
group=(
Gset
ML_Graph
ML_Graph
Bai
ML_Graph
GHS_indef
Goodwin
CPM
AG-Monien
VDOL
GHS_indef
ML_Graph
Pothen
Nasa
Schenk_IBMNA
DIMACS10
Lourakis
Bindel
Bindel
Bindel
Bindel
Bindel
Bindel
Pajek
Mallya
Mallya
Arenas
Schenk_IBMNA
Gaertner
TSOPF
Chen
Boeing
SNAP
Rajat
DIMACS10
Schenk_ISEI
Mallya
Mallya
HB
ML_Graph
DIMACS10
Bodendiek
Cunningham
Cunningham
Schenk_IBMNA
DIMACS10
Rommes
IBM_Austin
SNAP
vanHeukelum
Oberwolfach
Oberwolfach
Oberwolfach
SNAP
Hohn
Hohn
Rommes
Oberwolfach
Gaertner
Hollinger
Hollinger
SNAP
HB
GHS_indef
FIDAP
SNAP
GHS_indef
Bomhof
Mycielski
Cote
FEMLAB
GHS_indef
GHS_indef
Pothen
Nasa
GHS_indef
Schenk_IBMNA
Gaertner
Rommes
Rommes
Rommes
Rommes
Rommes
Rommes
Rommes
Pajek
Pajek
Grund
FEMLAB
Garon
VDOL
TKK
TKK
Hollinger
Hollinger
Grund
Boeing
Boeing
Boeing
HB
Simon
TSOPF
DIMACS10
Engwirda
Mallya
Mallya
Belcastro
QY
TSOPF
TSOPF
Averous
Hollinger
ACUSIM
Rajat
Schenk_IBMNA
Rommes
Schenk_IBMNA
TSOPF
IPSO
HB
HB
GHS_psdef
VDOL
Grund
Pothen
Nasa
LiuWenzhuo
VDOL
UTEP
Simon
VDOL
Andrianov
DIMACS10
DIMACS10
Hohn
Hohn
GHS_indef
GHS_psdef
FIDAP
Newman
Newman
Andrianov
Gupta
GHS_psdef
DIMACS10
Chen
Rommes
VDOL
Grund
PARSEC
GHS_psdef
Shen
Oberwolfach
Oberwolfach
Oberwolfach
VDOL
GHS_indef
Pothen
Mallya
Mallya
VDOL
Hollinger
Hollinger
Hamm
Botonakis
Goodwin
AG-Monien
ND
DNVS
VDOL
VDOL
Hollinger
Hollinger
Schenk_IBMNA
VDOL
Pothen
VDOL
VDOL
Shen
Schenk_ISEI
Pothen
TSOPF
GHS_psdef
SNAP
Andrianov
AG-Monien
Andrianov
Rajat
Pothen
FIDAP
Simon
PARSEC
Oberwolfach
Oberwolfach
JGD_Trefethen
GHS_indef
JGD_Trefethen
TAMU_SmartGridCenter
ML_Graph
Oberwolfach
Oberwolfach
AG-Monien
Oberwolfach
Oberwolfach
Oberwolfach
Oberwolfach
TSOPF
FEMLAB
CPM
Grund
Hollinger
Hollinger
Rajat
DNVS
Rommes
Szczerba
Rommes
FEMLAB
Rommes
Schenk_IBMNA
Simon
TKK
TKK
AG-Monien
VanVelzen
VanVelzen
VanVelzen
VanVelzen
DIMACS10
Chen
DNVS
Chen
Belcastro
Schenk_IBMNA
DIMACS10
Goodwin
SNAP
Oberwolfach
Li
Li
VanVelzen
VanVelzen
VanVelzen
VanVelzen
VanVelzen
VanVelzen
Newman
GHS_indef
Boeing
Boeing
Boeing
Andrianov
SNAP
Schenk_IBMNA
Pajek
Pajek
Puri
Puri
Puri
Puri
Bai
Hollinger
Hollinger
Schenk_IBMNA
AG-Monien
DIMACS10
GHS_indef
Rajat
Mycielski
Boeing
Boeing
Boeing
SNAP
HVDC
GHS_indef
DIMACS10
DIMACS10
Sandia
Sandia
Sandia
Averous
Boeing
Boeing
TSOPF
TKK
Wang
Wang
SNAP
Pajek
Hollinger
Hollinger
Hollinger
Hollinger
GHS_indef
Guettel
Schenk_IBMSDS
SNAP
Pajek
GHS_indef
TSOPF
TSOPF
HB
Schenk_IBMSDS
GHS_indef
FEMLAB
TSOPF
TSOPF
Dehghani
Hollinger
Hollinger
DNVS
IPSO
Andrianov
POLYFLOW
GHS_indef
GHS_indef
GHS_indef
GHS_indef
Rajat
Schenk_IBMNA
Boeing
Boeing
AG-Monien
GHS_psdef
POLYFLOW
TSOPF
Newman
SNAP
Schenk_IBMNA
Gupta
DIMACS10
GHS_indef
Andrianov
Goodwin
DIMACS10
DIMACS10
AG-Monien
DIMACS10
Quaglino
GHS_indef
PARSEC
Grund
Zhao
Zhao
Chen
Hollinger
Hollinger
SNAP
DNVS
Mallya
Mallya
GHS_indef
Hollinger
Hollinger
HB
TSOPF
Schenk_IBMNA
ND
ATandT
ATandT
Williams
GHS_psdef
AG-Monien
GHS_psdef
Nasa
Pothen
Hollinger
Hollinger
SNAP
SNAP
Andrianov
Chen
Rajat
Chevron
GHS_psdef
GHS_indef
Mancktelow
Schenk_IBMNA
TSOPF
TSOPF
TSOPF
Kim
GHS_indef
ML_Graph
Simon
vanHeukelum
DIMACS10
Rajat
GHS_psdef
GHS_psdef
GHS_psdef
DIMACS10
QY
Rommes
Rommes
Watson
Newman
TSOPF
GHS_psdef
Grueninger
CPM
Vavasis
GHS_indef
Hollinger
Hollinger
Hollinger
Hollinger
GHS_indef
Schenk_IBMNA
TSOPF
FEMLAB
Chen
Andrianov
Schenk_IBMNA
Schenk_IBMNA
IPSO
GHS_indef
HB
DIMACS10
Belcastro
DIMACS10
DIMACS10
Rothberg
Hollinger
Hollinger
Boeing
Boeing
Bova
VLSI
GHS_psdef
Hollinger
Hollinger
Schenk_IBMNA
Ronis
DIMACS10
GHS_psdef
Mycielski
QCD
AG-Monien
AG-Monien
AG-Monien
QCD
QCD
QCD
QCD
QCD
QCD
GHS_indef
IBM_EDA
IBM_EDA
Schenk_IBMNA
Schenk_IBMNA
GHS_psdef
GHS_indef
GHS_indef
Rajat
Schenk_IBMNA
Schenk_IBMNA
Schenk_IBMSDS
Schenk_IBMSDS
GHS_indef
Sanghavi
Boeing
Boeing
Pajek
GHS_psdef
Hollinger
Hollinger
Rothberg
Schenk_IBMSDS
Nasa
GHS_psdef
Hollinger
Hollinger
GHS_psdef
GHS_indef
Chen
Goodwin
DIMACS10
TSOPF
Grund
Schenk_IBMNA
Schenk_IBMNA
SNAP
DIMACS10
Hollinger
Hollinger
Andrews
GHS_indef
DIMACS10
GHS_indef
GHS_indef
TSOPF
Rudnyi
PARSEC
PARSEC
DIMACS10
Gupta
Simon
Simon
Simon
Williams
GHS_indef
SNAP
AG-Monien
Chen
GHS_psdef
Hollinger
Hollinger
GHS_indef
UTEP
DIMACS10
DIMACS10
DIMACS10
Cunningham
Cunningham
Andrianov
Oberwolfach
PARSEC
Raju
GHS_indef
DIMACS10
GHS_indef
Muite
Hamm
GHS_indef
LAW
DIMACS10
TAMU_SmartGridCenter
Mallya
Mallya
Rothberg
Koutsovasilis
ND
GHS_psdef
Lee
Rost
Mulvey
Mulvey
GHS_indef
VLSI
SNAP
TSOPF
Shyy
GHS_indef
DIMACS10
SNAP
SNAP
DIMACS10
Oberwolfach
Oberwolfach
Oberwolfach
Oberwolfach
GHS_indef
GHS_indef
Bomhof
GHS_indef
Chen
GHS_psdef
SNAP
MaxPlanck
MaxPlanck
SNAP
SNAP
Schmid
Pajek
Williams
GHS_indef
TSOPF
DIMACS10
Averous
Pothen
FEMLAB
DIMACS10
DIMACS10
Rajat
Rajat
Rajat
GHS_indef
Chen
LeGresley
GHS_indef
Andrianov
DIMACS10
Castrillon
Chevron
TKK
GHS_psdef
GHS_psdef
GHS_indef
CEMW
Pothen
Rajat
Rajat
Rajat
Chen
Chen
Embree
PARSEC
DNVS
Mycielski
Sandia
Sandia
DIMACS10
DIMACS10
DIMACS10
DIMACS10
Goodwin
GHS_psdef
DIMACS10
Um
Botonakis
Botonakis
Schenk_IBMSDS
Hamm
Oberwolfach
DNVS
Norris
Rajat
DIMACS10
Watson
PARSEC
PARSEC
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
PARSEC
DIMACS10
DNVS
DIMACS10
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Norris
Norris
IBM_EDA
IBM_EDA
IBM_EDA
IBM_EDA
IBM_EDA
VLSI
TSOPF
ATandT
Williams
DNVS
Rothberg
Pajek
Schenk_IBMSDS
Gleich
Oberwolfach
Gleich
vanHeukelum
DIMACS10
DIMACS10
DIMACS10
SNAP
DIMACS10
PARSEC
DIMACS10
DIMACS10
DIMACS10
DNVS
JGD_Homology
GHS_psdef
DIMACS10
TKK
DIMACS10
DIMACS10
UTEP
Botonakis
GHS_psdef
DIMACS10
AMD
Chen
Guettel
Schenk_ISEI
Rothberg
PARSEC
VLSI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
Schenk_ISEI
AG-Monien
Ronis
DIMACS10
SNAP
QLi
QLi
Fluorem
DIMACS10
DIMACS10
FreeFieldTechnologies
Schenk_IBMNA
Schenk_IBMNA
DIMACS10
Hamm
DIMACS10
Goodwin
Freescale
DNVS
GHS_indef
Guettel
Schenk_ISEI
DIMACS10
GHS_indef
PARSEC
DIMACS10
HVDC
GHS_indef
DIMACS10
DIMACS10
SNAP
DIMACS10
SNAP
Mycielski
DIMACS10
PowerSystem
DNVS
JGD_Homology
JGD_Homology
DIMACS10
DNVS
Botonakis
Botonakis
DIMACS10
VLSI
Williams
Norris
DNVS
DIMACS10
DIMACS10
Boeing
GHS_psdef
PARSEC
VLSI
DNVS
IPSO
IPSO
Bodendiek
DIMACS10
GHS_indef
DIMACS10
DIMACS10
PARSEC
Pajek
DIMACS10
BenElechi
DIMACS10
DIMACS10
Lin
Norris
DIMACS10
Um
SNAP
# DIMACS10
# DIMACS10
# DIMACS10
# Rajat
# SNAP
# DIMACS10
# PARSEC
# DIMACS10
# DIMACS10
# Meng
# Kamvar
# SNAP
# DIMACS10
# DIMACS10
# DIMACS10
# Sinclair
# DIMACS10
# Precima
# Andrianov
# SNAP
# Sandia
# Sandia
# Sorensen
# LAW
# SNAP
# Barabasi
# LAW
# DIMACS10
# SNAP
# DIMACS10
# Koutsovasilis
# Schenk_IBMNA
# DIMACS10
# Rajat
# DIMACS10
# Janna
# Chevron
# Fluorem
# GHS_indef
# GHS_indef
# GHS_indef
# VLSI
# Mycielski
# Tromble
# Martin
# SNAP
# SNAP
# SNAP
# Rajat
# Freescale
# INPRO
# Janna
# Dziekonski
# DIMACS10
# DIMACS10
# QLi
# vanHeukelum
# DIMACS10
# DIMACS10
# SNAP
# Kim
# GHS_indef
# DIMACS10
# Lee
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# GHS_psdef
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Schenk_AFE
# Mazaheri
# DIMACS10
# DIMACS10
# DIMACS10
# Williams
# Wissgott
# Andrianov
# DIMACS10
# Dziekonski
# Dziekonski
# Janna
# Rajat
# Rajat
# ATandT
# Sinclair
# Sandia
# Sandia
# Kamvar
# SNAP
# DIMACS10
# Chevron
# GHS_psdef
# CEMW
# LAW
# Janna
# Mycielski
# Bodendiek
# Gleich
# LAW
# DIMACS10
# Oberwolfach
# SNAP
# CEMW
# CEMW
# Janna
# Hardesty
# GHS_psdef
# GHS_psdef
# Oberwolfach
# McRae
# McRae
# Williams
# DIMACS10
# DIMACS10
# AG-Monien
# DIMACS10
# DIMACS10
# Schenk
# VLSI
# SNAP
# Dziekonski
# SNAP
# LAW
# SNAP
# Dziekonski
# VLSI
# Bodendiek
# Schmid
# Bourchtein
# Bourchtein
# LAW
# Janna
# SNAP
# Janna
# DIMACS10
# Hamrle
# VLSI
# Janna
# Janna
# Janna
# Bourchtein
# Bourchtein
# Janna
# Janna
# vanHeukelum
# Schenk_AFE
# Janna
# AMD
# Janna
# SNAP
# Gleich
# VLSI
# SNAP
# SNAP
# SNAP
# Fluorem
# Zaoui
# DIMACS10
# DIMACS10
# DIMACS10
# DIMACS10
# VLSI
# Janna
# Janna
# DIMACS10
# Bodendiek
# SNAP
# SNAP
# Freescale
# Janna
# Gleich
# Freescale
# Freescale
# SNAP
# Gleich
# Freescale
# DIMACS10
# Freescale
# Schenk
# Gleich
# DIMACS10
# Pajek
# SNAP
# DIMACS10
# SNAP
# DIMACS10
# Janna
# DIMACS10
# DIMACS10
# DIMACS10
# VLSI
# DIMACS10
# Rajat
# DIMACS10
# SNAP
# vanHeukelum
# LAW
# Freescale
# DIMACS10
# DIMACS10
# DIMACS10
# DIMACS10
# DIMACS10
# LAW
# DIMACS10
# Schenk
# DIMACS10
# DIMACS10
# Gleich
# VLSI
# DIMACS10
# DIMACS10
# DIMACS10
# DIMACS10
# DIMACS10
# Schenk
# DIMACS10
# DIMACS10
# DIMACS10
# LAW
# MAWI
# DIMACS10
# DIMACS10
# LAW
# DIMACS10
# GAP
# Schenk
# Sybrandt
# MAWI
# LAW
# LAW
# SNAP
# GAP
# LAW
# DIMACS10
# GenBank
# GAP
# SNAP
)

# suc=(
# 8
# 15
# 23
# 26
# 32
# 39
# 45   
# )

TARGET_DIR="./eval_data"

if [ -d "$TARGET_DIR" ]; then
   echo "'$TARGET_DIR' found and now copying files, please wait ..."
else
   echo "Warning: '$TARGET_DIR' NOT found. Create it"
   mkdir $TARGET_DIR
fi

OUTPUT=paperdata/large_V32N2M8.csv
rm -rf $OUTPUT
# rm -rf ${TARGET_DIR}/*

make clean && make spmm

for (( i=0; i<${#filename[@]}; i++ ));
# for i in "${suc[@]}"
do
echo "${group[$i]}/${filename[$i]}"
wget https://suitesparse-collection-website.herokuapp.com/MM/${group[$i]}/${filename[$i]}.tar.gz
tar zxvf ${filename[$i]}.tar.gz
mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
rm -rf ${filename[$i]}.tar.gz
rm -rf ${filename[$i]}

echo "./spmm --mtxfile ${TARGET_DIR}/${filename[$i]}.mtx --outmtxfile outmtx/tmp.mtx --maxiter 10 --n 64 --sched 0 --v 32"
printf "${filename[$i]}, " >> ${OUTPUT}
./spmm --mtxfile ${TARGET_DIR}/${filename[$i]}.mtx --outmtxfile outmtx/tmp.mtx --maxiter 10 --n 64 --sched 0 --v 32 >> ${OUTPUT}

# echo >> $OUTPUT
rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
done