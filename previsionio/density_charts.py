from selenium import webdriver
import uuid
import math
import time
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument("--test-type")
options.add_argument("--headless")
options.binary_location = "/usr/bin/chromium-browser"
driver = webdriver.Chrome(chrome_options=options)

best_round_frac = {1.5: 1, 3: 2, 7: 5}
best_frac = {1: 1, 2: 2, 5: 5}


def nice_num(range, round=False,
             best_round_frac=best_round_frac,
             best_frac=best_frac):
    exponent = math.floor(math.log10(range))
    exponent = math.pow(10, exponent)
    frac = range / exponent
    if round:
        for threshold in sorted(best_round_frac.keys()):
            if frac < threshold:
                return best_round_frac[threshold] * exponent
        return 10 * exponent

    for threshold in sorted(best_frac.keys()):
        if frac < threshold:
            return best_frac[threshold] * exponent
    return 10 * exponent


def nice_scale(lower_bound, upper_bound, max_ticks=6):
    range = nice_num(range=upper_bound - lower_bound, round=False)
    tick_spacing = nice_num(range / (max_ticks - 1), round=False)
    if (tick_spacing <= 0):
        return {'tickAmount': 4, 'min': 0, 'max': 1}
    niceLowerBound = math.floor(lower_bound / tick_spacing) * tick_spacing
    if niceLowerBound is None:
        niceLowerBound = 0
    niceUpperBound = math.ceil(upper_bound / tick_spacing) * tick_spacing
    if niceUpperBound is None:
        niceUpperBound = 0
    return {
        'tickAmount': math.ceil(upper_bound / tick_spacing),
        'min': niceLowerBound,
        'max': niceUpperBound
    }


def nice_scale_from_array(array=[], max_ticks=6):
    lower = min(array)
    upper = max(array)
    return nice_scale(lower, upper, max_ticks)


analysis = {"chart": {"type": "area"},
            "stroke": {"curve": "smooth"},
            "series": [{"name": "negatives",
                        "data": [0.004659697458418972, 0.01460517100817531, 0.0426526009791384, 0.09019832824409814,
                                 0.21050285862706497, 0.45315507482578427, 0.8143263376991318, 1.2752242284037079,
                                 1.8031909439989982, 2.3462959549064615, 2.847033975753461, 3.27780513113452,
                                 3.613683417270247, 3.8531170107594135, 3.9615968986961123, 3.9299030085855176,
                                 3.7736327017240976, 3.488012606003242, 3.1194501367788714, 2.742767514660776,
                                 2.39988571998813, 2.119029847866563, 1.903716810636499, 1.7407928035445859,
                                 1.5923198144575064, 1.4586508437954484, 1.3184170765973118, 1.200968157906547,
                                 1.0787955259915782, 0.9646604579953524, 0.8658434925393853, 0.7856901705095932,
                                 0.7241174740196162, 0.6797852264662623, 0.6523500714617448, 0.6354036994484149,
                                 0.6231976131542027, 0.6287884097959164, 0.6328620234142666, 0.6361203884356047,
                                 0.6214868907149605, 0.5974607051572394, 0.5579341890017268, 0.5099631875016366,
                                 0.4530079473732198, 0.3910978449067697, 0.34146319096732475, 0.2992732732602832,
                                 0.26715763273620613, 0.23437344495835097, 0.2015484686822885, 0.17618695817159055,
                                 0.1544097366480323, 0.1376318459145031, 0.11946681524148077, 0.10352103130279806,
                                 0.09394361982061297, 0.08190090934093842, 0.06928199341395747, 0.05866286943636042,
                                 0.04964165691038968, 0.03997621903353302, 0.031574227470874704, 0.02629174196305228,
                                 0.024403668497711382, 0.021398163338381418, 0.017275226485062384, 0.012075383653072307,
                                 0.005725397619425359, 0.0017587613532206044, 0]
                        },
                       {"name": "positives",
                        "data": [0, 0.0001539346060914926, 0.005955370034355618, 0.010859919497707465,
                                 0.021369329660338374, 0.05898459338241044, 0.11849075069054223, 0.18799290279720549,
                                 0.2732234789114821, 0.3643758992651342, 0.4719153888136035, 0.5682881427771261,
                                 0.6399764785120003, 0.6982307425050257, 0.728999758024732, 0.7327532180168924,
                                 0.7233892817341359, 0.7033139420880232, 0.6681669275071868, 0.6333162752121939,
                                 0.6117741739489925, 0.6145049400959999, 0.6340409678284634, 0.6455451185630452,
                                 0.6668721299834713, 0.6940317979086191, 0.7247968769315751, 0.7591897640316823,
                                 0.7981134036153418, 0.8354308294746007, 0.8719717187660616, 0.9159735388486284,
                                 0.9650258116106283, 1.0079561768340386, 1.0387419507806541, 1.0913500947415453,
                                 1.152853064194047, 1.2198773286245006, 1.2714422268037984, 1.3280079012404449,
                                 1.386768252970979, 1.4280568082008347, 1.4931437290267369, 1.5626561847307252,
                                 1.638689796042941, 1.7028919618832683, 1.7903681576132586, 1.8815575658962838,
                                 1.950454491476658, 2.007499279229049, 2.053589256256055, 2.1206806765154567,
                                 2.1854186033929452, 2.232821561049675, 2.254812377429947, 2.2292917736101088,
                                 2.1822168615852644, 2.0869982928005735, 1.9460809975374345, 1.7518869824834253,
                                 1.551594991459026, 1.3334636411918455, 1.1013070136247571, 0.8685049144140304,
                                 0.6465260828101957, 0.4602360308339725, 0.31289798584411854, 0.19156075334706862,
                                 0.1031384245031104, 0.04598696652412173, 0.019456329952225706]
                        }
                       ],
            "xaxis": {"categories": [0, 0.0143, 0.0286, 0.0429, 0.0571,
                                     0.0714, 0.0857, 0.1, 0.1143, 0.1286,
                                     0.1429, 0.1571, 0.1714, 0.1857, 0.2,
                                     0.2143, 0.2286, 0.2429, 0.2571, 0.2714,
                                     0.2857, 0.3, 0.3143, 0.3286, 0.3429,
                                     0.3571, 0.3714, 0.3857, 0.4, 0.4143,
                                     0.4286, 0.4429, 0.4571, 0.4714, 0.4857,
                                     0.5, 0.5143, 0.5286, 0.5429, 0.5571,
                                     0.5714, 0.5857, 0.6, 0.6143, 0.6286,
                                     0.6429, 0.6571, 0.6714, 0.6857, 0.7,
                                     0.7143, 0.7286, 0.7429, 0.7571, 0.7714,
                                     0.7857, 0.8, 0.8143, 0.8286, 0.8429,
                                     0.8571, 0.8714, 0.8857, 0.9, 0.9143,
                                     0.9286, 0.9429, 0.9571, 0.9714, 0.9857, 1]
                      }
            }
data = []
for d in analysis["series"]:
    data.append(d["data"])

nice_scale_dict = nice_scale_from_array(data[0])

options_dict = '''
{
   annotations:{
      xaxis:[
         {
            x:0.5,
            strokeDashArray:0,
            borderColor:'#e55539',
            label:{
               text:undefined,
               borderWidth:0,
               style:{
                  background:'#e55539',
                  color:'#FFF'
               }
            }
         }
      ]
   },
   chart:{ type:'area', height:'auto',
            toolbar:{
                  tools:{
                     selection:false,
                     zoom:false,
                     zoomin:false,
                     zoomout:false,
                     pan:false,
                     reset:false
                  }
               }
   },
   colors: ['#e55539', '#77cacf'],
   dataLabels:{ enabled:false},
   stroke:{ curve:'smooth', width:1.5 },
   series:[
      {
         name: "negatives",
         data:''' + (str(analysis["series"][0]["data"])) + '''
      },
      {
         name: "positives",
         data:''' + (str(analysis["series"][1]["data"])) + '''
      }
   ],
   xaxis:{
      type :'numeric',
      categories : ''' + (str(analysis['xaxis']['categories'])) + ''',
      labels:{
         formatter: function(value) {
            if (typeof value !== 'number') return value;
            return new Intl.NumberFormat('en', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                        }).format(value);
            }
        },
        min : 0,
        max : 1,
        tickAmount:4
   },

   yaxis:{
      tickAmount : ''' + (str(nice_scale_dict.get('tickAmount'))) + ''',
      min : ''' + (str(nice_scale_dict.get('min'))) + ''',
      max : ''' + (str(nice_scale_dict.get('max'))) + ''',
      labels : {
         formatter: function(value) {
            if (typeof value !== 'number') return value;
            return new Intl.NumberFormat('en', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                        }).format(value);
                      }
                }
    }
}
'''

html = '''
<html>
  <body>
      <h1 style="font-family : sans-serif;color:#35495e;">Density</h1>
      <div id="chart"></div>
      <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
      <script>
          var options = {}
          var chart = new ApexCharts(document.querySelector("#chart"), options);
          chart.render();
      </script>

  </body>
</html>
'''.format(options_dict)

filename = str(uuid.uuid1()) + ".html"
with open("/tmp/" + filename, "w+") as f:
    f.write(html)

driver.get('file:///tmp/' + filename)
print('/tmp/' + filename)
time.sleep(1)
driver.save_screenshot("screenshot.png")
driver.close()
