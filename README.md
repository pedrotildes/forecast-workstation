# Forecast Workstation

Estação técnica de previsão numérica do tempo: mapas de alta qualidade com
camadas combináveis e sondagens aerológicas previstas (Skew-T log-p estilo
RAOB), a partir dos modelos **GFS**, **ECMWF IFS** e **ECMWF AIFS**.

## Arrancar

```bash
./run.sh
```

Abre em <http://localhost:8600>. Na primeira execução cria `.venv` (Python 3.12)
e instala `requirements.txt`.

## Fontes de dados (GRIB2 originais, sem APIs intermédias)

| Modelo | Fonte | Resolução | Runs / alcance |
|---|---|---|---|
| GFS | NOMADS `grib_filter` (recorte por região/variável/nível) | 0.25°, 23 níveis de pressão | 00/06/12/18Z, +384 h (horário até +120 h) |
| ECMWF IFS | ECMWF Open Data (pedidos HTTP por intervalo de bytes via `.index`), espelho AWS | 0.25°, 13 níveis | 00/12Z +360 h; 06/18Z +90 h |
| ECMWF AIFS | idem (`aifs-single`) | 0.25°, 13 níveis | 4 runs/dia, +360 h, passo 6 h |
| ICON-EU (DWD) | opendata.dwd.de (um ficheiro `.grib2.bz2` por campo/nível/passo) | 0.0625° (~7 km), 18 níveis | 00/06/12/18Z +120 h (horário até +78 h); 03/09/15/21Z +48 h |

O ICON-EU é **regional** (23,5° W–62,5° E, 29,5° N–70,5° N): cobre Portugal continental e a
Madeira, mas não os Açores nem as Canárias — a app avisa quando um ponto ou região fica fora.
O servidor do DWD guarda só as últimas ~24 h. Produtos de longo alcance (termograma,
persistência de calor/frio, meteograma comparativo) usam a run mais recente **completa**
de cada modelo, para que uma run ainda em publicação não os encurte.

Cada mensagem GRIB é guardada individualmente em `cache/grib/…` e
descodificada com ecCodes. As imagens geradas ficam em `cache/img/`. A cache é
limpa automaticamente (mantém as 3 runs mais recentes de cada modelo e imagens
com menos de 2 dias). Os pedidos ao NOMADS são limitados a ~97/min (o NOMADS
bloqueia acima de 120/min).

Pontos (sondagens e meteogramas) usam "tiles" fixos de 4° + 1° de margem: pontos
próximos partilham os mesmos downloads.
Os dados ECMWF estão sob licença CC BY 4.0 (atribuição incluída nos mapas).

## Arquitetura

```
backend/app/
  models/      fontes NWP com vocabulário canónico comum (gfs.py, ecmwf.py, base.py)
  grib.py      descodificação ecCodes → grelha lat/lon normalizada
  products.py  catálogo de variáveis (brutas e derivadas: θe, vorticidade,
               advecção, espessura, K, TT, cisalhamento, precip. acumulada…)
  regions.py   domínios e projeções (Lambert/Mercator)
  maps/        styles.py (paletas e isolinhas), compose.py (presets → job),
               render.py (Cartopy/Matplotlib)
  sounding/    profile.py (perfil no ponto), analysis.py (MetPy), skewt.py, compare.py
  series.py    séries temporais num ponto (todos os passos de uma run)
  meteogram.py meteograma individual e multimodelo
  maps/diff.py mapas de diferenças entre modelos
  climate.py   normais diárias Tmáx/Tmín (CPC 1991–2020)
  daily.py     Tmáx/Tmín diárias a partir das janelas de cada modelo
  heatwave.py  critérios de vaga de calor/frio, mapas e sinal do ENS
  thermogram.py termograma de um ponto
  main.py      API FastAPI; renderização num pool de processos
frontend/      index.html + app.js + style.css (sem build)
```

## API

- `GET /api/models`, `/api/models/{id}/runs`, `/api/regions`, `/api/catalog?model=`
- `GET /api/map?model&run&step&region&preset=` ou `&layers=[…]` (PNG; cabeçalho
  `X-Map-Axes` dá a posição do mapa na imagem)
- `GET /api/locate?region&fx&fy` → lat/lon
- `GET /api/sounding.png|json?model&run&step&lat&lon&parcel=SB|ML|MU`
- `GET /api/meteogram.png?model&lat&lon[&run&hours=240&profile=true&name]` — meteograma
  (T/Td, precipitação, nebulosidade, vento/rajadas, MSLP, CAPE; secção tempo-altura no GFS)
- `GET /api/compare/times?models=gfs,ecmwf,aifs` — horas válidas comuns (run/passo por modelo)
- `GET /api/compare/meteogram.png?models&lat&lon&hours` — meteograma multimodelo
- `GET /api/compare/sounding.png?models&valid=AAAAMMDDHH&lat&lon` — Skew-T sobreposto
- `GET /api/compare/diff.png?a&b&var&level&valid&region[&accum]` — mapa de diferenças A − B
- `GET /api/progress?job=…` — progresso de descargas longas (meteogramas)
- `GET /api/extremes/point.png?lat&lon&name` — termograma de vagas de calor/frio
- `GET /api/extremes/map.png?kind=heat|cold|tmax_anom|tmin_anom&model&region&d0&d1`
- `GET /api/extremes/ens.png?param=p_gt1p5…&region[&step | &d0&d1]` — sinal ECMWF ENS
- `GET /api/climate/status` — estado da climatologia CPC

## Satélite e radar (separador Satélite/Radar)

Mapa interativo (Leaflet) com imagens pedidas diretamente pelo browser:

- **Satélite** — EUMETSAT EUMETView (WMS): MTG-I1 (FCI, a cada 10 min: IR 10,5 µm,
  VIS 0,6 µm, GeoColour, cor real, tipo/fase de nuvem, nevoeiro, poeiras) e MSG
  (SEVIRI, a cada 15 min: vapor de água 6,2 µm, massa de ar, convecção, microfísica,
  cor natural, altura do topo das nuvens).
- **Sobreposições** — radar de precipitação (mosaico RainViewer, 10 min, últimas ~2 h,
  zoom nativo até 7), descargas elétricas (MTG Lightning Imager), RDT (trovoadas em
  desenvolvimento rápido), precipitação estimada por satélite (H SAF), costa/fronteiras.
- Animação das últimas 1–12 h; clique num ponto para abrir sondagem, meteograma ou
  termograma.
- Limites dos fornecedores: EUMETView ~20 pedidos/s e 8 simultâneos (a app usa no
  máximo 3 e repete pedidos recusados); RainViewer 500 mosaicos/min (mosaicos de 512 px
  e só 3 imagens de radar carregadas de cada vez).
- `GET /api/live/layers` — camadas disponíveis e hora da última imagem.

## Vagas de calor e de frio

- Tmáx/Tmín diárias (UTC): Tmáx = períodos centrados entre 06–18 UTC; Tmín = 18–06 UTC.
  GFS usa TMAX/TMIN (janelas de 6 h), ECMWF IFS `mx2t3/mn2t3` (`mx2t6` após +144 h),
  AIFS só T2m de 6 em 6 h (subestima ligeiramente os extremos).
- Normais diárias: NOAA CPC Global Unified Temperature 1991–2020 (0,5°, só sobre terra),
  descarregadas automaticamente na primeira utilização (~550 MB → recorte de 19 MB em
  `cache/clim/`).
- Critério IPMA/OMM: ≥ 6 dias consecutivos com Tmáx ≥ normal + 5 °C (calor) ou
  Tmín ≤ normal − 5 °C (frio).
- ECMWF ENS: probabilidade (51 membros) de anomalia padronizada de T850 > ±1/1,5/2σ
  face ao clima do modelo, de 12 em 12 h até +360 h (runs 00/12Z).

Exemplo de `layers`:

```json
[{"type":"fill","var":"t","level":850},
 {"type":"fill","var":"precip","accum":6},
 {"type":"contour","var":"msl","interval":2},
 {"type":"wind","level":850,"style":"barbs"}]
```

## Versões para computador (macOS e Windows)

A app também é distribuída como aplicação de computador: o servidor e a interface são
empacotados com **PyInstaller** e abertos numa janela nativa (**pywebview**).

- `backend/desktop.py` — ponto de entrada: arranca o servidor numa porta local livre
  e abre a janela. `FORECAST_HEADLESS=1` arranca só o servidor (testes).
- `backend/forecast.spec` — especificação do PyInstaller (inclui ecCodes, Cartopy,
  PROJ, MetPy, h5py e a interface).
- Na app empacotada os dados ficam na pasta do utilizador
  (`~/Library/Application Support/ForecastWorkstation` no Mac,
  `%LOCALAPPDATA%\ForecastWorkstation` no Windows).
- Build local no Mac:

  ```bash
  cd backend && ../.venv/bin/python -m PyInstaller forecast.spec --noconfirm --clean
  ```

- **CI** (`.github/workflows/build-desktop.yml`): ao enviar uma tag `v*`
  (ex.: `git tag v1.0.1 && git push origin v1.0.1`) o GitHub Actions constrói macOS
  (Apple Silicon) e Windows, testa o arranque de cada um e publica os zips numa
  *Release*.
- Os executáveis não estão assinados: no Mac abrir com botão direito → Abrir; no
  Windows "Mais informações → Executar mesmo assim" (ver `docs/LEIA-ME.txt`).
