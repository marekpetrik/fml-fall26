# Builds the eigen-climates dataset: the average temperature on each day of the
# year for 100 cities in the USA and Canada.
#
# Source: NOAA Global Historical Climatology Network Daily (GHCN-Daily), which
# includes both US and Canadian weather stations.
#
# Run: julia download-climate.jl   (takes a few minutes)

using Downloads, CSV, DataFrames, Dates, Statistics

const FIRST_YEAR = 1991
const LAST_YEAR  = 2020
const DIR = @__DIR__
const RAW = joinpath(DIR, "raw")
mkpath(RAW)

### --- Cities

cities = CSV.read(IOBuffer("""
city,region,country,latitude,longitude
St. John's,NL,Canada,47.56,-52.71
Halifax,NS,Canada,44.65,-63.58
Charlottetown,PE,Canada,46.24,-63.13
Fredericton,NB,Canada,45.96,-66.64
Quebec City,QC,Canada,46.81,-71.21
Montreal,QC,Canada,45.50,-73.57
Ottawa,ON,Canada,45.42,-75.70
Toronto,ON,Canada,43.65,-79.38
Windsor,ON,Canada,42.31,-83.04
Sudbury,ON,Canada,46.49,-80.99
Thunder Bay,ON,Canada,48.38,-89.25
Winnipeg,MB,Canada,49.90,-97.14
Churchill,MB,Canada,58.77,-94.17
Regina,SK,Canada,50.45,-104.62
Saskatoon,SK,Canada,52.13,-106.67
Edmonton,AB,Canada,53.55,-113.49
Calgary,AB,Canada,51.05,-114.07
Kamloops,BC,Canada,50.67,-120.33
Vancouver,BC,Canada,49.28,-123.12
Victoria,BC,Canada,48.43,-123.37
Prince George,BC,Canada,53.92,-122.75
Prince Rupert,BC,Canada,54.31,-130.32
Whitehorse,YT,Canada,60.72,-135.06
Yellowknife,NT,Canada,62.45,-114.37
Inuvik,NT,Canada,68.36,-133.72
Iqaluit,NU,Canada,63.75,-68.52
Durham,NH,USA,43.13,-70.93
Jackson,NH,USA,44.14,-71.18
Portland,ME,USA,43.66,-70.26
Caribou,ME,USA,46.86,-68.01
Burlington,VT,USA,44.48,-73.21
Boston,MA,USA,42.36,-71.06
New York,NY,USA,40.71,-74.01
Albany,NY,USA,42.65,-73.75
Buffalo,NY,USA,42.89,-78.88
Philadelphia,PA,USA,39.95,-75.17
Pittsburgh,PA,USA,40.44,-80.00
Washington,DC,USA,38.90,-77.04
Richmond,VA,USA,37.54,-77.44
Raleigh,NC,USA,35.78,-78.64
Asheville,NC,USA,35.60,-82.55
Charleston,SC,USA,32.78,-79.93
Atlanta,GA,USA,33.75,-84.39
Jacksonville,FL,USA,30.33,-81.66
Tampa,FL,USA,27.95,-82.46
Miami,FL,USA,25.76,-80.19
Key West,FL,USA,24.56,-81.78
Birmingham,AL,USA,33.52,-86.80
Nashville,TN,USA,36.16,-86.78
Memphis,TN,USA,35.15,-90.05
Louisville,KY,USA,38.25,-85.76
New Orleans,LA,USA,29.95,-90.07
Little Rock,AR,USA,34.75,-92.29
Chicago,IL,USA,41.88,-87.63
Detroit,MI,USA,42.33,-83.05
Marquette,MI,USA,46.55,-87.40
Cleveland,OH,USA,41.50,-81.69
Columbus,OH,USA,39.96,-83.00
Indianapolis,IN,USA,39.77,-86.16
Milwaukee,WI,USA,43.04,-87.91
Minneapolis,MN,USA,44.98,-93.27
Duluth,MN,USA,46.79,-92.10
International Falls,MN,USA,48.60,-93.41
Des Moines,IA,USA,41.59,-93.62
St. Louis,MO,USA,38.63,-90.20
Kansas City,MO,USA,39.10,-94.58
Omaha,NE,USA,41.26,-95.93
Fargo,ND,USA,46.88,-96.79
Bismarck,ND,USA,46.81,-100.78
Rapid City,SD,USA,44.08,-103.23
Oklahoma City,OK,USA,35.47,-97.52
Dallas,TX,USA,32.78,-96.80
Houston,TX,USA,29.76,-95.37
San Antonio,TX,USA,29.42,-98.49
El Paso,TX,USA,31.76,-106.49
Amarillo,TX,USA,35.22,-101.83
Brownsville,TX,USA,25.90,-97.50
Denver,CO,USA,39.74,-104.99
Albuquerque,NM,USA,35.08,-106.65
Phoenix,AZ,USA,33.45,-112.07
Flagstaff,AZ,USA,35.20,-111.65
Las Vegas,NV,USA,36.17,-115.14
Salt Lake City,UT,USA,40.76,-111.89
Boise,ID,USA,43.62,-116.20
Billings,MT,USA,45.78,-108.50
Missoula,MT,USA,46.87,-113.99
Cheyenne,WY,USA,41.14,-104.82
Seattle,WA,USA,47.61,-122.33
Spokane,WA,USA,47.66,-117.43
Portland,OR,USA,45.52,-122.68
Eureka,CA,USA,40.80,-124.16
San Francisco,CA,USA,37.77,-122.42
Sacramento,CA,USA,38.58,-121.49
Fresno,CA,USA,36.74,-119.79
Los Angeles,CA,USA,34.05,-118.24
San Diego,CA,USA,32.72,-117.16
Honolulu,HI,USA,21.31,-157.86
Anchorage,AK,USA,61.22,-149.90
Fairbanks,AK,USA,64.84,-147.72
Juneau,AK,USA,58.30,-134.42
Utqiagvik,AK,USA,71.29,-156.79
Pittburgh,NH,USA,45.18,-71.12
Amherst,MA,USA,42.37,-72.51
Mt Washington,NH,USA,44.30,-71.31
Death Valley,CA,USA,36.67,-117.20
"""), DataFrame)

### --- Download (or reuse) a file

function download_cached(url, path)
    isfile(path) && return path
    for attempt in 1:3
        try
            Downloads.download(url, path * ".part")
            return mv(path * ".part", path)
        catch e
            attempt == 3 && rethrow()
            println("  download failed, retrying: ", e)
            sleep(10)
        end
    end
end

### --- Station list and inventory (which years each station reports TMAX and TMIN)

ghcn = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
inventory_path = download_cached("$ghcn/ghcnd-inventory.txt", joinpath(RAW, "ghcnd-inventory.txt"))
stations_path  = download_cached("$ghcn/ghcnd-stations.txt",  joinpath(RAW, "ghcnd-stations.txt"))

# fixed-width format, see https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
station_names = Dict(strip(l[1:11]) => strip(l[42:min(71, end)])
             for l in eachline(stations_path) if startswith(l, r"US|CA"))

years  = Dict{String,Vector{Int}}()                 # id => years with TMAX and TMIN
coords = Dict{String,Tuple{Float64,Float64}}()     # id => (latitude, longitude)
for l in eachline(inventory_path)
    startswith(l, r"US|CA") || continue
    element = l[32:35]
    element in ("TMAX", "TMIN") || continue
    id = l[1:11]
    coords[id] = (parse(Float64, l[13:20]), parse(Float64, l[22:30]))
    # first and last year of TMAX and TMIN: [tmax_first, tmax_last, tmin_first, tmin_last]
    y = get!(years, id, [typemax(Int), typemin(Int), typemax(Int), typemin(Int)])
    k = element == "TMAX" ? 1 : 3
    y[k], y[k+1] = parse(Int, l[37:40]), parse(Int, l[42:45])
end

# number of years in [FIRST_YEAR, LAST_YEAR] in which the station reports both TMAX and TMIN
function overlap(y)
    first = max(y[1], y[3], FIRST_YEAR)
    last  = min(y[2], y[4], LAST_YEAR)
    return max(0, last - first + 1)
end

stations = DataFrame(id = collect(keys(years)))
stations.lat = [coords[id][1] for id in stations.id]
stations.lon = [coords[id][2] for id in stations.id]
stations.overlap = [overlap(years[id]) for id in stations.id]
filter!(r -> r.overlap > 0, stations)
println("Stations with TMAX and TMIN in $FIRST_YEAR-$LAST_YEAR: ", nrow(stations))

### --- Helpers

function distance_km(lat1, lon1, lat2, lon2)      # great-circle (haversine) distance
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    a = sin((φ2 - φ1) / 2)^2 + cos(φ1) * cos(φ2) * sin(deg2rad(lon2 - lon1) / 2)^2
    return 2 * 6371 * asin(sqrt(a))
end

# stations near a city to download; only stations in the same country (ID prefix US or CA)
# so that border cities, such as Windsor and Detroit, do not share stations
function candidates(lat, lon, country; radius = 30, max_stations = 8)
    prefix = country == "Canada" ? "CA" : "US"
    c = filter(r -> startswith(r.id, prefix), stations)
    c.distance = distance_km.(lat, lon, c.lat, c.lon)
    filter!(r -> r.distance ≤ radius, c)
    sort!(c, [order(:overlap, by = o -> o < 20), :distance])
    return first(c, max_stations)
end

# daily TMAX and TMIN (°C) from the NOAA Access Data Service
function daily_data(ids, path)
    url = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries" *
          "&stations=$(join(ids, ","))" *
          "&startDate=$FIRST_YEAR-01-01&endDate=$LAST_YEAR-12-31" *
          "&dataTypes=TMAX,TMIN&units=metric&format=csv&includeAttributes=false"
    download_cached(url, path)
    df = CSV.read(path, DataFrame; types = Dict(:STATION => String, :DATE => Date,
                                                :TMAX => Float64, :TMIN => Float64))
    return dropmissing(df, [:TMAX, :TMIN])
end

# one record per day: the station with the most days of data among those with data that day
function splice(df, ids)
    counts = combine(groupby(df, :STATION), nrow => :n)
    sort!(counts, [order(:n, rev = true), order(:STATION, by = s -> findfirst(==(s), ids))])
    rank = Dict(s => i for (i, s) in enumerate(counts.STATION))
    df = transform(df, :STATION => ByRow(s -> rank[s]) => :rank)
    sort!(df, [:DATE, :rank])
    return unique(df, :DATE)
end

# average temperature on each day of a non-leap year
function climatology(daily)
    daily = filter(r -> !(month(r.DATE) == 2 && day(r.DATE) == 29), daily)
    daily.doy = dayofyear.(Date.(2001, month.(daily.DATE), day.(daily.DATE)))
    daily.tavg = (daily.TMAX .+ daily.TMIN) ./ 2
    c = sort(combine(groupby(daily, :doy), :tavg => mean => :tavg, nrow => :n), :doy)
    nrow(c) == 365 || error("missing days of the year")
    return c
end

### --- Download and process each city

ndays = Dates.value(Date(LAST_YEAR, 12, 31) - Date(FIRST_YEAR, 1, 1)) + 1
slug(s) = replace(lowercase(s), r"[^a-z0-9]+" => "-")

climate = zeros(nrow(cities), 365)
used = DataFrame(city = String[], region = String[], coverage = Float64[],
                 min_years = Int[], stations = String[])

for (i, c) in enumerate(eachrow(cities))
    for radius in (30, 60)
        cand = candidates(c.latitude, c.longitude, c.country; radius)
        nrow(cand) == 0 && continue
        path = joinpath(RAW, "$(slug(c.city))-$(lowercase(c.region))-$(radius)km.csv")
        daily = splice(daily_data(cand.id, path), cand.id)
        coverage = nrow(daily) / ndays
        clim = coverage ≥ 0.8 ? climatology(daily) : nothing
        if clim === nothing || minimum(clim.n) < 15
            println("$(c.city), $(c.region): coverage $(round(100coverage, digits=1))% " *
                    "within $radius km is not enough")
            continue
        end
        climate[i, :] = clim.tavg
        share = combine(groupby(daily, :STATION), nrow => :n)
        sort!(share, :n, rev = true)
        desc = join(["$(s.STATION) $(station_names[s.STATION]) ($(round(Int, 100 * s.n / nrow(daily)))%)"
                     for s in eachrow(share)], "; ")
        push!(used, (c.city, c.region, coverage, minimum(clim.n), desc))
        println("$(c.city), $(c.region): $(round(100coverage, digits=1))% of days from ",
                nrow(share), " station(s)")
        break
    end
    nrow(used) == i || error("no usable stations for $(c.city), $(c.region)")
end

### --- Save

out = copy(cities)
for d in 1:365
    out[!, "d" * lpad(d, 3, '0')] = round.(climate[:, d], digits = 1)
end
CSV.write(joinpath(DIR, "climate.csv"), out)
CSV.write(joinpath(DIR, "stations.csv"), used)
println("Saved ", joinpath(DIR, "climate.csv"), ": ", size(out))
