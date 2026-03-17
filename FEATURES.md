# Final Feature Stack

This file documents the exact final feature stack used by the best FloodGraphFlow-XGB submission path.

- Base engineered features before pruning: `231
- Appended Stage A stacked features: `31`
- Final regressor input dimension: `262`

## Base Features (231)

1. `rain_sum_w1`
2. `rain_sum_w2`
3. `rain_sum_w4`
4. `rain_sum_w8`
5. `rain_sum_w16`
6. `rain_sum_w32`
7. `rain_sum_w64`
8. `rain_max_w1`
9. `rain_max_w2`
10. `rain_max_w4`
11. `rain_max_w8`
12. `rain_max_w16`
13. `rain_max_w32`
14. `rain_max_w64`
15. `rain_ewm_a0p9`
16. `rain_ewm_a0p97`
17. `rain_ewm_a0p99`
18. `warm_wl_first`
19. `warm_wl_last`
20. `warm_wl_mean`
21. `warm_wl_std`
22. `warm_wl_slope`
23. `warm_dwl_last`
24. `warm_d2wl_last`
25. `warm_dwl_abs_mean_short`
26. `warm_dwl_abs_mean_long`
27. `warm_dwl_burst_ratio`
28. `position_x`
29. `position_y`
30. `depth`
31. `invert_elevation`
32. `surface_elevation`
33. `base_area`
34. `rel_elev_local_vs_neighbors`
35. `sink_depth_rel_neighbors`
36. `cap_fullness_1d`
37. `cap_headroom_1d`
38. `cap_surface_gap_1d`
39. `cap_pond_depth_2d`
40. `cap_pond_norm_2d`
41. `cap_near_1d_0p8`
42. `cap_near_1d_0p9`
43. `cap_near_2d_0p8`
44. `cap_near_2d_0p9`
45. `pipe_has_outflow`
46. `pipe_min_outflow_diameter`
47. `pipe_mean_outflow_diameter`
48. `t_rel`
49. `t_rel_norm`
50. `lead_bin_0_35`
51. `lead_bin_36_71`
52. `lead_bin_72_107`
53. `lead_bin_108_223`
54. `lead_bin_224_plus`
55. `rain_burst_ratio_sum`
56. `rain_burst_ratio_max`
57. `rain_since_last`
58. `rain_since_peak`
59. `future_rain_sum_1`
60. `future_rain_max_1`
61. `future_rain_sum_2`
62. `future_rain_max_2`
63. `future_rain_sum_4`
64. `future_rain_max_4`
65. `future_rain_sum_8`
66. `future_rain_max_8`
67. `future_rain_sum_16`
68. `future_rain_max_16`
69. `future_rain_sum_32`
70. `future_rain_max_32`
71. `future_rain_sum_64`
72. `future_rain_max_64`
73. `endpoint_is_inlet_1d`
74. `endpoint_is_outfall_1d`
75. `endpoint_is_boundary_1d`
76. `endpoint_inlet_x_future_rain8`
77. `endpoint_inlet_x_future_rain24`
78. `endpoint_inlet_x_future_rain32`
79. `endpoint_outfall_x_ds_lock`
80. `endpoint_outfall_x_interface_drain_pressure`
81. `endpoint_boundary_x_drain_gate`
82. `endpoint_boundary_x_late_gate`
83. `endpoint_boundary_x_tnorm`
84. `endpoint_inlet_x_future_rain64`
85. `endpoint_inlet_x_rain_ewm_0p99`
86. `endpoint_inlet_x_basin_mass_deficit_ratio`
87. `endpoint_inlet_x_node_prior_q95`
88. `routing_rain_lag_1`
89. `routing_rain_lag_2`
90. `routing_rain_lag_4`
91. `routing_rain_lag_8`
92. `routing_rain_lag_12`
93. `routing_rain_lag_24`
94. `routing_rain_lag_36`
95. `routing_rain_lag_48`
96. `routing_rain_dr1`
97. `routing_rain_slope4`
98. `routing_rain_slope8`
99. `routing_since_global_peak`
100. `routing_since_last_rain`
101. `routing_cum_since_calm`
102. `routing_area_x_rain_now`
103. `routing_area_x_rain_short`
104. `routing_flow_accumulation_x_rain_now`
105. `routing_flow_accumulation_x_rain_short`
106. `routing_centroid_elevation_x_rain_now`
107. `routing_centroid_elevation_x_rain_short`
108. `upstream_rain_sum_w4`
109. `upstream_rain_mean_w4`
110. `upstream_rain_hop2_sum_w4`
111. `upstream_rain_hop2_mean_w4`
112. `upstream_rain_sum_w8`
113. `upstream_rain_mean_w8`
114. `upstream_rain_hop2_sum_w8`
115. `upstream_rain_hop2_mean_w8`
116. `upstream_rain_sum_w16`
117. `upstream_rain_mean_w16`
118. `upstream_rain_hop2_sum_w16`
119. `upstream_rain_hop2_mean_w16`
120. `upstream_rain_sum_w32`
121. `upstream_rain_mean_w32`
122. `upstream_rain_hop2_sum_w32`
123. `upstream_rain_hop2_mean_w32`
124. `interaction_rain_now_x_warm_slope`
125. `interaction_rain_now_x_warm_abs_dy_mean_long`
126. `interaction_depth_x_rain_now`
127. `interaction_invert_elevation_x_rain_now`
128. `interaction_surface_elevation_x_rain_now`
129. `hydro_up_cap`
130. `hydro_dn_cap`
131. `hydro_dn_cap_min`
132. `hydro_up_flow`
133. `hydro_dn_flow`
134. `hydro_up_wl_mean`
135. `hydro_pressure_mean_1d2d`
136. `hydro_fill_pct`
137. `hydro_freeboard`
138. `hydro_base_area`
139. `hydro_area`
140. `hydro_flow_accumulation`
141. `fillrain_rain_now_x_fill_pct`
142. `fillrain_rain_short_x_fill_pct`
143. `fillrain_future_rain16_x_fill_pct`
144. `fillrain_rain_now_div_freeboard`
145. `fillrain_rain_short_div_freeboard`
146. `fillrain_future_rain16_div_freeboard`
147. `tx_rain_now_fill_log1p`
148. `tx_rain_short_fill_log1p`
149. `tx_future_rain4_fill_log1p`
150. `tx_rain_now_div_free_log1p`
151. `tx_rain_short_div_free_log1p`
152. `tx_future_rain16_div_free_log1p`
153. `tx_dvdh_area_ratio_log1p`
154. `tx_store_delta_asinh`
155. `tx_qhist_fast_qin_upsum_asinh`
156. `tx_qhist_slow_qin_upsum_asinh`
157. `tx_qhist_fast_qout_upsum_asinh`
158. `tx_targeted_freeboard_warm_asinh`
159. `tx_freeboard_asinh`
160. `tx_headroom_1d_asinh`
161. `tx_dvdh_abs_dh_asinh`
162. `tx_node_wl_over_basin_mean_stable_asinh`
163. `zae_base_zero`
164. `zae_endpoint_zero`
165. `zae_base_zero_x_release`
166. `zae_base_zero_x_unlock`
167. `zae_endpoint_zero_x_rain8`
168. `zae_endpoint_zero_x_rain32`
169. `zae_endpoint_zero_x_freeboard`
170. `targeted_indeg_1d`
171. `targeted_outdeg_1d`
172. `targeted_up_len_1d`
173. `targeted_dn_len_1d`
174. `targeted_sink_depth_2d`
175. `targeted_is_surcharged_warm`
176. `targeted_freeboard_warm`
177. `targeted_lag8_rain_x_up_len_1d`
178. `targeted_has_sink_depth`
179. `dvdh_live_area`
180. `dvdh_area_ratio`
181. `dvdh_abs_dh`
182. `global_outlet_hops_norm`
183. `global_elev_above_outfall`
184. `basin_mean_wl_lag9`
185. `basin_rain24`
186. `node_wl_over_basin_mean`
187. `twi_proxy`
188. `spi_proxy`
189. `slope_proxy`
190. `log1p_area_proxy`
191. `routingpot_outlet_hops_norm`
192. `routingpot_upstream_area_frac`
193. `routingpot_hierarchy_proxy`
194. `routingpot_inout_ratio`
195. `routingpot_travel_impedance`
196. `routingpot_node_len_mean`
197. `hand_surface`
198. `hand_centroid`
199. `hand_invert`
200. `hand_nearest_drain_inv`
201. `hand_wl_minus_nearest_drain_inv`
202. `basin_mass_deficit`
203. `basin_mass_deficit_ratio`
204. `basin_storage_now`
205. `downstream_full_frac`
206. `downstream_headroom_mean`
207. `downstream_wl_mean`
208. `downstream_lock_pressure`
209. `subcatch_mass_deficit`
210. `subcatch_mass_deficit_ratio`
211. `subcatch_storage_now`
212. `mass_node_deficit`
213. `mass_up1_deficit_mean`
214. `mass_up2_deficit_mean`
215. `mass_node_minus_basin_deficit`
216. `mass_up1_minus_basin_deficit`
217. `mass_up2_minus_basin_deficit`
218. `node_prior_max_wl`
219. `node_prior_q95_wl`
220. `node_prior_flood_freq`
221. `drain_prior_dry_drop_q50`
222. `drain_prior_dry_drop_q90`
223. `drain_prior_dry_drop_rate`
224. `drain_prior_dry_recession_slope`
225. `uphist_ema_fast_upsum`
226. `uphist_ema_slow_upsum`
227. `uphist_ema_fast_upmean`
228. `uphist_ema_slow_upmean`
229. `uphist_ema_fastslow_ratio`
230. `uphist_ema_outfall_slow`
231. `interaction_base_area_x_rain_now`

## Stage A Appended Features (31)

232. `qnet_pred`
233. `qnet_cumsum`
234. `qnet_rollsum_3`
235. `qnet_rollsum_8`
236. `qnet_rollsum_24`
237. `qnet_lag_1`
238. `qnet_lag_2`
239. `qnet_lag_4`
240. `qhat_upstream_sum`
241. `qhat_upstream_mean`
242. `qhat_up_down_imbalance`
243. `qhat_upstream_hop2_sum`
244. `qhat_upstream_hop2_mean`
245. `qnet_phys_baseline_wl`
246. `qin_pred`
247. `qout_pred`
248. `qin_cumsum`
249. `qin_rollsum_3`
250. `qin_rollsum_8`
251. `qin_rollsum_24`
252. `qin_lag_1`
253. `qin_lag_2`
254. `qin_lag_4`
255. `qout_cumsum`
256. `qout_rollsum_3`
257. `qout_rollsum_8`
258. `qout_rollsum_24`
259. `qout_lag_1`
260. `qout_lag_2`
261. `qout_lag_4`
262. `aux_peak_within_24`

## Derivation Map

### Warm-start rainfall history

- Derived from the observed warm-start rainfall sequence only.
- `rain_sum_*` are rolling sums.
- `rain_max_*` are rolling maxima.
- `rain_ewm_*` are exponentially weighted moving summaries.

### Warm-start water-level summaries

- `warm_wl_*` summarize the observed warm-start water-level history.
- `warm_d*` summarize first and second differences over the same warm-start window.

### Static and geometry context

- `position_x`, `position_y`, `depth`, `invert_elevation`, `surface_elevation`, and `base_area` are direct static inputs.
- `rel_elev_*` compare a node’s elevation to its undirected neighbors.
- `cap_*` encode fullness, headroom, surface gap, and 1D/2D near-capacity indicators.
- `pipe_*` are static 1D outflow bottleneck descriptors.
- `global_*`, `twi_*`, `spi_*`, `routingpot_*`, and `hand_*` are static topology/terrain/drainage proxies.

### Time and forcing context

- `t_rel`, `t_rel_norm`, and `lead_bin_*` condition the model on forecast horizon.
- `rain_burst_*` capture burstiness and recency.
- `future_rain_*` are leak-safe known rainfall forcing windows.
- `endpoint_*` specialize forcing and lockup context for inlet/outfall/boundary nodes.
- `routing_*` encode rainfall lags, rise-rate, recency, and a few routing-style static interactions.
- `upstream_rain_*` aggregate 1-hop and 2-hop upstream rain summaries.

### Interaction features

- `interaction_*` are simple nonlinear rain-by-state or rain-by-static products.
- `fillrain_*` combine rainfall with fill ratio and freeboard.

### Hydraulic state and physics-aware features

- `hydro_*` encode local/upstream/downstream hydraulic context from warm-start state, conveyance, and geometry.
- `targeted_*` are hand-built topology and surcharge-sensitive physics proxies.
- `dvdh_*` approximate local storage-response behavior.
- `basin_mean_wl_lag9`, `basin_rain24`, and `node_wl_over_basin_mean` provide basin-scale context.

### Mass-balance features

- `basin_mass_deficit*` are basin-scale storage deficit proxies.
- `subcatch_mass_deficit*` are the same idea at subcatchment scale.
- `mass_*deficit*` compare node, 1-hop upstream, 2-hop upstream, and basin imbalance levels.
- These are the main mass-balance-style features that drove the strong mid-branch score drop.

### Priors and historical regime features

- `node_prior_*` are train-split node priors from historical maximum water level, q95 water level, and exceedance frequency.
- `drain_prior_*` are train-split dry/recession priors.
- `uphist_ema_*` summarize upstream warm-start history with fast and slow EMA channels.

### Transform cleanup v3

- `tx_*` features are stabilized `log1p` or `asinh` transforms of otherwise brittle hydraulic ratios or interactions.
- `zae_*` features explicitly handle zero-area endpoint edge cases.
- This cleanup block is the main feature difference between final public `Model_2` and `Model_1`.

### Stage A stacked channels

- `qnet_*` are out-of-fold net-flow surrogate predictions plus cumulative, rolling, and lag summaries.
- `qhat_*` summarize those predicted net flows over the drainage graph.
- `qin_*` and `qout_*` are out-of-fold inflow and outflow surrogate predictions plus state summaries.
- `aux_peak_within_24` is the appended auxiliary event target prediction.
