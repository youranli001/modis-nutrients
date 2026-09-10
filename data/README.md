# data/

Not versioned (see `.gitignore`). Expected here:

- `satellite_and_WOA13_1_degree_Jan_v2.nc` — 1° January grid with the seven
  MODIS inputs (`CHL APH FLU PIC POC PAR SST`), the three WOA13 surface targets
  (`nitrate phosphate silicate`) and a `depth` field. 5 MB. Produced by the
  workshop notebooks from the sources below.

Sources:

- MODIS-Aqua L3m monthly climatology, 9 km: https://oceancolor.gsfc.nasa.gov/l3/
  (files `AQUA_MODIS.<start>_<end>.L3m.MC.<PRODUCT>.9km.nc`)
- WOA13 v2 nutrients, 1°, monthly: https://www.ncei.noaa.gov/data/oceans/woa/WOA13/DATAv2/
  (files `woa13_all_{n,p,i}<MM>_01.nc`, variables `n_an`, `p_an`, `i_an`)

Layout expected by `modis-nutrients download` / `build`:

```
data/raw/woa13/<nitrate|phosphate|silicate>/woa13_all_<n|p|i><MM>_01.nc
data/raw/modis/<CHL|APH|FLU|PIC|POC|PAR|SST>/AQUA_MODIS.<start>_<end>.L3m.MC.<SUITE>.<var>.9km.nc
data/satellite_and_WOA13_1deg_monthly.nc      # output of `build`
```
