# Doppler, propagation and inference specification

## Coordinates and time

WGS84 geodetic latitude, longitude and ellipsoid height are converted with EPSG:4979
-> EPSG:4978, then rotated into east/north/up about a declared origin. Euclidean
slant ranges are computed in this frame; the transform preserves Earth-centered
straight-line distance. A planar candidate surface z=0 is a tangent plane, not a DEM.
For real terrain use a raster and explicit vertical datum. Orthometric heights H require
ellipsoid heights h=H+N, with geoid separation N. A single configured N is a local
approximation. Barometric height requires its own atmospheric correction and must not
be silently treated as ellipsoid height, MSL height, or altitude above ground.

Trajectory times t are seconds since an aware UTC origin. Acoustic annotation times u
are relative to recording start, with one shared offset Δ: t_a=u+Δ. PCHIP interpolation
provides continuous piecewise differentiable 3D P(t); V(t)=dP/dt is taken from the same
interpolant. Long gaps are segmented, not bridged. Inputs must remain subsonic throughout
the evaluated interval. Interpolation error and ADS-B timestamp errors are not assumed zero.

## Moving source, fixed observer

Let r(t_e)=P(t_e)-O, d=||r|| and let the approach radial velocity be

```
v_r = -r · V / d
f_obs(t_e) = f0 * c / (c - v_r)
t_a(t_e) = t_e + d(t_e)/c
```

Because dd/dt_e=r·V/d=-v_r,

```
dt_a/dt_e = 1 - v_r/c
f_obs/f0 = dt_e/dt_a = 1/(1-v_r/c).
```

Approaching sources have positive v_r and higher observed frequency. Receding sources
have lower frequency. The production model samples emission time, computes arrival time
and ratio, verifies monotonicity, then interpolates the ratio onto requested arrival times.
It never plots emission-time frequency as though it occurred at the microphone at emission.
Out-of-support arrival times return NaN; scoring rejects incomplete event coverage.
The default 0.25 s emission step is configurable through `predict`; convergence should be
checked for fast turns or rapid acceleration.

Constant c assumes homogeneous still air. A configurable dry-air approximation is
c=sqrt(γ R T_K), γ=1.4 and R=287.05 J/(kg K). Humidity, pressure profile, wind and refraction
are omitted. Configuring another c is sensitivity analysis, not atmospheric ray tracing.
Supersonic sources are rejected: a monotonic single-arrival model is not a sonic-boom model.

## Closest point of approach

For straight P=P0+Vt, unconstrained CPA is

```
t_CPA = -(P0-O)·V / (V·V)
```

clipped to observed support. General paths use a one-second distance sweep plus bounded
refinement of local minima, checking endpoints. Delayed CPA is t_CPA+d(t_CPA)/c.
It is the arrival time of sound emitted at CPA, not necessarily the peak energy or midpoint
of an observed curve. A one-second screen can miss tightly spaced local minima; finer
sampling is needed for unusually aggressive maneuvers.

## Scoring and shared alignment

For event i, harmonic order h_i, ratio q_ij(O,Δ), observed frequencies y_ij and uncertainty
σ_i, the expected frequency is h_i f0_i q_ij. The Gaussian objective is the mean squared
normalized residual. Unknown f0_i is profiled analytically:

```
f0_i = clip(sum(h_i q_ij y_ij) / sum((h_i q_ij)^2), f_min, f_max)
L_i = mean(((y_ij - h_i f0_i q_ij)/σ_i)^2)
L_joint = mean_i L_i
```

Events are weighted equally so an oversampled annotation does not dominate merely by
having more points. This corresponds to a chosen composite error model, not a claim that
correlated STFT bins are independent. The robust loss replaces r² by 2(sqrt(1+r²)-1),
while profiling f0 still uses least squares; it is an approximate robust score, not an exact
robust maximum-likelihood estimator. Harmonic order is an explicit hypothesis; multiple
harmonics from one source should not be treated as independent aircraft.

Every event uses the same Δ grid. No event can independently slide to improve a match.
Arrival-support rejection provides temporal screening; delayed CPA values and event timing
are diagnostics. Absence of another aircraft contributes no negative evidence in this
version because a calibrated detectability model is unavailable.

Energy evidence samples log(1+power/median-frame-power) along predicted harmonic curves.
Stationary noise, interfering tones and chance bright ridges can yield high energy.
Energy is an optional component diagnostic, not aircraft identity or calibrated probability.

## Uncertainty and identifiability

Single straight flights often have mirror symmetry about the ground track and uncertain
along-track position/time. Multiple nonparallel flight events can improve identifiability,
but an incorrect association, time origin, tone or datum can still fit misleadingly well.
The near-optimal candidate radius describes only sampled solutions within loss+1;
it is NOT a confidence interval or calibrated probability.

A conditional perturbation ensemble resamples frequency errors and varies sound speed and
flight height. Samples describe sensitivity to those chosen assumptions. They omit systematic
source-tone errors, unknown winds, ADS-B selection bias and multimodal missed candidates.
Do not label their envelope a validated coverage region. Fit residuals >2 sigma (mean
normalized squared loss >4) trigger an insufficient-evidence warning.

## Independent synthetic generator

For analytic P(t), solve t_e+||P(t_e)-O||/c=t_a with Newton iterations. Evaluate source phase
φ_h(t_a)=2π h f0 t_e(t_a), sum harmonics, apply a smooth emission envelope, optional
geometric spreading and seeded Gaussian noise. No production `predict` routine supplies
this waveform. Reject Doppler-shifted harmonics above Nyquist, not just source tones.
Compare dφ/dt_a against the production model as an independent check.

The end-to-end demo extracts ridges from the noisy waveform. Known frequency-band and
active-time windows are explicit synthetic priors, not learned aircraft detection. The
observer truth is passed to the generator and evaluator only. Held-out geometry changes
observer location, source velocities, altitude, climb/descent and source frequencies.

## References

- NASA Glenn, [Doppler effect](https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/doppler.html): moving-source acoustic wavefront geometry.
- OpenStax, [Doppler effect and sonic booms](https://openstax.org/books/physics/pages/14-3-doppler-effect-and-sonic-booms): subsonic domain and limits.
- SciPy, [PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) and [STFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html): numerical component definitions.
- PROJ, [cartesian conversion](https://proj.org/en/stable/operations/conversions/cart.html): geodetic/geocentric geometry.

The linked inspiration video is not a specification of its author's unpublished code.
The equations and architecture here are independently derived and implemented.
