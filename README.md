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
| ERA5 (reanálise ECMWF/C3S) | NCAR RDA ds633.0 por OPeNDAP (recorte por região, nível e hora), só para eventos históricos | 0.25°, 27 níveis até 100 hPa | horária (análises de 3 em 3 h no arquivo) |

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
  xsection.py  cortes verticais ao longo de uma linha A→B (θ, θe, HR, ω, PV de Ertel)
  series.py    séries temporais num ponto (todos os passos de uma run)
  meteogram.py meteograma individual e multimodelo
  maps/diff.py mapas de diferenças entre modelos
  climate.py   normais diárias Tmáx/Tmín (CPC 1991–2020)
  daily.py     Tmáx/Tmín diárias a partir das janelas de cada modelo
  heatwave.py  critérios de vaga de calor/frio, mapas e sinal do ENS
  thermogram.py termograma de um ponto
  main.py      API FastAPI; renderização num pool de processos
frontend/      index.html + app.js + xsect.js + live.js + style.css (sem build)
```

## API

- `GET /api/models`, `/api/models/{id}/runs`, `/api/regions`, `/api/catalog?model=`
- `GET /api/map?model&run&step&region&preset=` ou `&layers=[…]` (PNG; cabeçalho
  `X-Map-Axes` dá a posição do mapa na imagem)
- `GET /api/locate?region&fx&fy` → lat/lon
- `GET /api/sounding.png|json?model&run&step&lat&lon&parcel=SB|ML|MU`
- `GET /api/xsection.png?model&run&step&lat0&lon0&lat1&lon1&kind=dyn|normal|moist|omega|pv|temp[&a&b]`
  — corte vertical previsto de A para B (`/api/xsection/kinds` lista os tipos)
- `GET /api/meteogram.png?model&lat&lon[&run&hours=240&profile=true&name]` — meteograma
  (T/Td, precipitação, nebulosidade, vento/rajadas, MSLP, CAPE; secção tempo-altura no GFS)
- `GET /api/compare/times?models=gfs,ecmwf,aifs` — horas válidas comuns (run/passo por modelo)
- `GET /api/compare/meteogram.png?models&lat&lon&hours` — meteograma multimodelo
- `GET /api/compare/sounding.png?models&valid=AAAAMMDDHH&lat&lon` — Skew-T sobreposto
- `GET /api/compare/diff.png?a&b&var&level&valid&region[&accum]` — mapa de diferenças A − B
- `GET /api/compare/xsection.png?models&valid&lat0&lon0&lat1&lon1&kind[&diff=1]` — cortes verticais
  de vários modelos à mesma hora (painéis empilhados; opcionalmente diferença 1.º − 2.º modelo)
- `GET /api/progress?job=…` — progresso de descargas longas (meteogramas)
- `GET /api/extremes/point.png?lat&lon&name` — termograma de vagas de calor/frio
- `GET /api/extremes/map.png?kind=heat|cold|tmax_anom|tmin_anom&model&region&d0&d1`
- `GET /api/extremes/ens.png?param=p_gt1p5…&region[&step | &d0&d1]` — sinal ECMWF ENS
- `GET /api/climate/status` — estado da climatologia CPC

## Cortes verticais

No separador **Sondagens → Corte vertical A→B** (ou em **Mapas**, com «Clique no mapa abre: Corte
vertical», clicando em A e depois em B). A linha define-se por linhas predefinidas (Madeira→Lisboa,
Açores→Lisboa, Lisboa→Madrid, paralelo 39°N…), por coordenadas, ou no mapa pequeno (clique ou
arraste os marcadores). O corte segue o círculo máximo, usa todos os níveis de pressão do modelo
até 100 hPa (GFS 21, ICON-EU 18, ECMWF/AIFS 12) e a barra de tempo percorre os passos da run.

| Tipo | Sombreado | Linhas |
|---|---|---|
| Vento, θ e tropopausa | velocidade do vento (kt) + barbelas | θ, 2 PVU (tropopausa dinâmica) |
| Vento normal ao corte | componente normal (⊗ para dentro = à esquerda de A→B) | θ, 2 PVU |
| θe, humidade e subida | HR ≥ 50 % + barbelas | θe, ω < 0 |
| Movimento vertical | ω (Pa/s; subida a vermelho) | θ, HR 90 % |
| Vorticidade potencial | PV de Ertel isobárica (PVU) | θ, isotácas ≥ 60 kt, 2 PVU |
| Temperatura | T (°C) + barbelas | 0 °C, HR 90 % |

Em **Comparar → Corte vertical**, a mesma linha é cortada em cada modelo assinalado à mesma hora
válida (cada um na run mais recente que a cobre), em painéis empilhados com a mesma escala de cores e
um só mapa de localização; opcionalmente, um painel com a diferença entre os dois primeiros modelos
(campo sombreado do tipo de corte, nos níveis comuns aos dois).

A PV é calculada na grelha (campos suavizados a ~25 km), com todos os termos da forma isobárica
(vorticidade absoluta × ∂θ/∂p e termos baroclínicos); no ICON-EU o ω é suavizado a ~0,2°. Cada
figura inclui um mapa de localização (pressão MSL + jato a 300 hPa ou θe a 850 hPa), a superfície
(pressão à superfície) e a escala de altitude da atmosfera padrão.

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

## Eventos (arquivo permanente)

Separador **Eventos**: define-se um evento (nome, janela UTC, modelos, antecedência das
previsões) e a app copia para um arquivo permanente — recortado à região "Atlântico NE e
Ibéria" (42° W–14° E, 24°–52° N: Açores, Madeira, continente) e quantizado a 16 bits — os
campos necessários para todos os produtos:

- **sequência de análises**: todas as runs 00/06/12/18Z iniciadas durante o evento (passo +0 h
  com todos os níveis de pressão, para sondagens; +3 h com os campos principais);
- **previsões**: as runs 00Z/12Z desde N dias antes, de 6 em 6 h dentro da janela
  (superfície + 850/700/500/250 hPa).
- **cadeia de curto prazo**: primeiras 6 h de cada run (pressão, vento, rajada, precipitação,
  T2m), horária no GFS e no ICON-EU;
- **radar** (RainViewer) de 10 em 10 min enquanto o evento decorre (mosaicos 512 px, z3–z6).

O arquivo fica em `~/Library/Application Support/ForecastWorkstation/archive`
(`%LOCALAPPDATA%\ForecastWorkstation\archive` no Windows; ou `FORECAST_ARCHIVE`), partilhado
pelo servidor de desenvolvimento e pela app empacotada, e a limpeza da cache não lhe toca.
`Model.fetch` recorre ao arquivo quando o servidor já não tem os dados, e as runs
arquivadas aparecem nas listas com a indicação «arquivo». No separador Satélite/Radar, o
seletor "Período" anima o evento com o satélite EUMETSAT (arquivo de anos no EUMETView) e o
radar arquivado.

Recuperação posterior: GFS (AWS `noaa-gfs-bdp-pds`; NOMADS ~10 dias) e ECMWF IFS/AIFS
(espelho AWS `ecmwf-forecasts`) podem ser arquivados depois; **ICON-EU (~24 h) e radar
(~2 h) só enquanto o evento decorre**. Com a app fechada, o arquivador pode correr sozinho:

```bash
cd backend && ../.venv/bin/python -m app.archive
```

### Eventos históricos

Um evento que já acabou há mais de 10 dias é **histórico**: os servidores de previsão já não
têm os dados, por isso o arquivo é preenchido a partir de outras fontes, sem conta:

- **ERA5** (`app/era5.py`): reanálise do ECMWF lida da cópia do NCAR RDA (ds633.0) por OPeNDAP,
  só para a caixa do evento — 27 níveis (gh, T, HR, u, v, ω) de 3 em 3 h, superfície horária
  (MSL, pressão à superfície, vento a 10 m, T2m, Td2m, nebulosidade, água precipitável, CAPE) e
  precipitação/rajadas das previsões curtas (06/18 UTC), arrumadas como a «cadeia de curto prazo».
  Aparece como modelo **ERA5 (reanálise)** em todos os separadores.
- **Análises do GFS** (`app/gfs_hist.py`): ficheiros `gfsanl_4` (0,5°, 6 em 6 h) do arquivo da
  NOAA/NCEI, para eventos anteriores a março de 2021 (as previsões dessa época não estão em
  acesso aberto).
- **Previsões arquivadas na AWS**: se o evento começa depois de 01/03/2021 (GFS,
  `noaa-gfs-bdp-pds`), 20/01/2023 (ECMWF IFS, espelho `ecmwf-forecasts`) ou 01/03/2025 (AIFS),
  as análises e as previsões desses modelos são descarregadas como num evento atual, por
  intervalos de bytes. O GFS usa o NOMADS nos últimos 9 dias e a AWS antes disso; o ECMWF usa o
  data.ecmwf.int nos últimos 4 dias e o espelho da AWS antes disso.
- **NHC ATCF** (`app/atcf.py`): para ciclones tropicais (ex.: `AL132018`), a trajetória observada
  (best track) e as previsões oficiais e dos modelos; a chegada a terra é o primeiro ponto em terra
  (Natural Earth 50 m). No separador Casos: «Previsões NHC: trajetórias» e «… chegada a terra».
- **Radiossondas** antigas: Univ. Wyoming (também com o número WMO antigo da estação, p. ex.
  Lisboa 08579) e NOAA IGRA v2 (ficheiro do período completo).

A região de um evento pode ser «Atlântico NE e Ibéria» ou «Atlântico NE, Ibéria e Ilhas
Britânicas» (para tempestades que seguem para norte, como o Ophelia em 2017). Os cortes verticais
da ERA5 aceitam também as análises intermédias (de 3 em 3 h), e os mapas da ERA5 dizem
«Análise» em vez de «Run/Previsão».

## Casos (análise de casos particulares)

Separador **Casos**, sobre um evento arquivado:

- **Trajetórias** — cada sistema (semente: nome, posição e hora, p. ex. do comunicado do IPMA) é
  seguido nas análises de cada modelo como mínimo fechado da pressão MSL suavizada (3/3 h; AIFS 6/6 h):
  em cada passo, o mínimo mais próximo da posição esperada pelo movimento, admitindo reformações do
  centro (até ~950 km na direção do movimento) e ignorando a margem dos dados de cada modelo. As
  trajetórias previstas são as das runs 00/12Z de −24/−48/−72/−96 h antes do momento de referência
  do sistema (opcional, p. ex. a chegada a terra; por omissão, o de pressão mínima); fundo: pressão
  mínima atingida durante o evento.
- **Pressão central** — analisada por modelo vs. prevista.
- **Sequência sinótica** — mosaico de pressão MSL + precipitação em 6 h a cada 6/12/24 h.
- **Precipitação total** e **rajada máxima** do evento — a partir da *cadeia de curto prazo*
  (primeiras 6 h de cada run de 6 em 6 h; GFS/ICON-EU horário, ECMWF 3 h, AIFS 6 h).
- **Evolução num local** — pressão, vento/rajada, intensidade da precipitação e T2m.
- **Sondagens** — radiossonda observada (preto) e análises dos modelos (passo +0 h) sobrepostas,
  com hodógrafas e tabela de índices lado a lado (observado vs. modelos).
- **Meteograma do caso** — secção tempo-altura das análises de 6 em 6 h (HR, T, vento), com as horas
  das radiossondas da estação mais próxima, e os painéis de superfície da cadeia de curto prazo.
- **Corte vertical** — os mesmos cortes, sobre as análises arquivadas de 6 em 6 h, incluindo linhas
  «pelo centro» de cada sistema (O→E, S→N, SO→NE, NO→SE; 1600 km) que acompanham o centro
  analisado quando se muda de hora (◀ ▶).

Radiossondas observadas (`app/obs.py`): Univ. Wyoming (BUFR de alta resolução) e, em alternativa,
NOAA IGRA v2 (ficheiros *year-to-date*, 1–2 dias de atraso), para Lisboa, Funchal, Lajes, Corunha,
Santander, Madrid, Saragoça, Múrcia, Palma, Barcelona, Tenerife e Bordéus. O arquivador recolhe as
sondagens das 00/12Z de cada evento e guarda-as em `archive/obs/`.

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

## Serviço de avisos (Telegram)

`python -m app.alerts` (em `backend/`) segue, sem parar, os fenómenos extremos previstos para
Portugal continental, a Madeira e os Açores e manda um alerta ilustrado para o telemóvel quando
há **certeza grande** de chegar ao nível laranja ou vermelho dos critérios do IPMA.

- **Áreas**: os 18 distritos (todos os pontos da grelha de 0,25° dentro de cada distrito), as 4
  áreas da Madeira e os 3 grupos dos Açores (vilas e pontos do interior), com os códigos do IPMA.
- **Fenómenos e critérios** (`app/alerts/criteria.py`, página «Critérios de Emissão dos Avisos
  Meteorológicos» do IPMA): rajadas (91 km/h; 111 nas terras altas, acima de 700 m de
  orografia do modelo a 0,25°; Açores 101, ou vento médio 75 km/h), precipitação em 6 h (41 mm) e em 1 h (21 mm), tempo quente e frio (limiares
  de cada distrito durante ≥ 48 h = dois dias seguidos); vermelho: > 130 km/h, > 60 mm e os
  limiares vermelhos de cada distrito, > 40 mm em 1 h, vento médio > 87 km/h nos Açores; agitação marítima (altura significativa das ondas
  5 m, vermelho > 7 m; Açores 7 e 9 m; costa sul do Algarve com ondulação de sueste 3 e 5 m);
  neve acumulada por faixa de altitude (abaixo de 1000 m 6 cm, vermelho > 15; 1000–1600 m 11 e
  30 cm; acima de 1600 m 26 e 50 cm), em 24 h; trovoada (índice de tempestade severa, abaixo);
  nevoeiro persistente (≥ 72 h, vermelho ≥ 96 h, abaixo). São todos os critérios do IPMA.
- **Vento médio nos Açores** (`app/alerts/meanwind.py`): o IPMA avisa nos Açores também pela
  velocidade média (laranja 75–87 km/h, vermelho > 87). Entra no aviso de vento, como no IPMA: num
  grupo, um membro chega ao nível pela rajada ou pelo vento médio (vento a 10 m, de 3 em 3 h),
  o que for maior. O vento a 10 m custa ~1,7 MB por membro e passo, mas o vento do modelo nunca
  passa da rajada máxima da janela que acaba nesse passo: nos temporais de 19–20/01/2025,
  11/03/2025 e 23–24/02/2026 a menor razão rajada/vento foi 1,09 e um vento médio ≥ 75 km/h veio
  sempre com rajadas ≥ 93 km/h. Por isso só se descarrega para os membros com rajada ≥ 82 km/h
  num ponto dos Açores, e só nos dias em que pelo menos 20 % deles a têm num grupo. Na prática
  decide a rajada (20/01/2025, grupo Ocidental: 58 % de laranja pelo vento médio, 94 % pela
  rajada); o vento médio sobe sobretudo a probabilidade de vermelho (aí, de 4 % para 12 %).
- **Chuva em 1 h** (`app/alerts/rain1h.py`): o ensemble do ECMWF só publica a precipitação de
  3 em 3 h, por isso aqui o ensemble é o **ICON-EU-EPS do DWD** (40 membros, grelha icosaédrica
  com células a ~10 km, precipitação horária até +48 h, runs de 6 em 6 h; cada ficheiro traz os
  40 membros de um passo para a Europa, ~10 MB). Pontos: as células do ICON dentro de cada
  distrito (~50 por distrito) e, na Madeira, a célula mais próxima de cada vila. Os Açores ficam
  fora do domínio do ICON-EU: aí o ensemble é o **ICON-EPS global** da mesma run (40 membros,
  células a ~20 km, chuva horária até +48 h só nas runs das 00 e 12 UTC; a célula mais próxima
  de cada vila, todas a menos de 14 km). Um ficheiro global tem ~32 MB por passo, por isso só é
  lido quando o ECMWF dá ≥ 20 % de 25 mm em 24 h num ponto dos Açores. Triagem exata: a chuva é acumulada desde o início da run, por isso
  uma hora com ≥ 21 mm obriga a ≥ 21 mm no dia e nas 6 h que a contêm; primeiro lêem-se os totais
  do dia, depois os de 6 h, e só as janelas de 6 h em que algum membro chega ao limiar laranja
  descarregam os ficheiros horários (dia seco: 2 ficheiros). Modelos de apoio: os que têm chuva
  horária, ICON-EU e GFS (margem 8 mm). Até 2 dias. Limitação: uma célula de 10–28 km recebe a
  média da chuva; uma trovoada isolada pode dar 30–50 mm/h em poucos km² e a célula muito menos,
  por isso o serviço vê sobretudo a chuva forte organizada (linhas de instabilidade, frentes
  com convecção, depressões isoladas). Na tempestade Charles (11/11/2026, 12–13 UTC) o ICON-EU
  deu 22 mm/h e o GFS 17 mm/h no distrito de Lisboa, a muito curto prazo.
- **Trovoada** (`app/alerts/storms.py`): desde agosto de 2026 o IPMA define o aviso pelos
  fenómenos esperados — laranja: condições para fenómenos *severos* localizados (granizo ≥ 2 cm,
  rajadas ≥ 90 km/h, 20–40 mm em 1 h); vermelho: *extremos* (granizo grande, rajadas muito
  destrutivas, ≥ 40 mm em 1 h). Um modelo global a 9 km não faz granizo nem rajadas de
  trovoada, por isso o aviso lê-se no **ambiente** de cada membro do ensemble do ECMWF, como fazem
  os previsores: **√(2·MUCAPE) × cisalhamento do vento entre 925 e 500 hPa** (WMAXSHEAR, m²/s²),
  o parâmetro que melhor separa as trovoadas comuns das severas e estas das extremamente severas
  nas sondagens europeias (Taszarek et al. 2017, 2020; é o «CAPE-shear» do ECMWF sem o fator √2).
  Só conta onde o próprio membro dá precipitação (≥ 1 mm nas 6 h em volta) e com MUCAPE ≥ 500 J/kg:
  CAPE que não é libertada não faz trovoadas, e nas frentes de inverno o cisalhamento sozinho
  (30–40 m/s com 200–400 J/kg) chegaria ao limiar. O valor de um dia é o máximo nas 8 horas
  sinópticas. Limiares: laranja ≥ 900, vermelho ≥ 2000 m²/s² (nas sondagens, ~500–700 nas
  severas e ~1000 nas extremamente severas; aqui é o máximo de um distrito num dia, daí valores
  mais altos). Os dias são triados com o mesmo índice no IFS (≥ 60 % do limiar, ± 1 dia); o vento
  a 925 e 500 hPa (o grosso do download) só é descarregado para os membros que têm chuva e CAPE
  acima do mínimo nos pontos dos distritos a essa hora: poucos num dia de inverno, todos num dia
  de trovoadas generalizadas (~0,5 GB). Modelos de apoio: IFS, GFS e ICON-EU (este com a CAPE da camada de mistura), a 40 % do
  limiar. Até 2 dias. Calibração com o arquivo do ECMWF na AWS (run das 00 UTC do dia; o IPMA
  avaliava antes a frequência das descargas, por isso os casos até 2025 não medem a mesma coisa):
  10–11/06/2025, laranja do IPMA em 7 distritos do Norte e Centro — 100 % nesses 7, mas também
  70–100 % em Bragança, Guarda, Castelo Branco, Leiria, Santarém e Portalegre; 29/07/2024, laranja
  em 13 distritos — no máximo 60 % (Santarém); 7–8/06/2024, laranja em 8 — ≤ 20 %; 3 e
  5/10/2026, laranja subido pelo IPMA no próprio dia (Setúbal e Évora; Guarda, Castelo Branco e
  Santarém) — ≤ 10 % nesses distritos e ≤ 52 % fora deles; 5/09/2026, amarelo em Aveiro (houve
  granizo > 2 cm e um *downburst*) — ≤ 12 %; frentes de 11 e 21/11/2026 — ≤ 14 %; frente fria de
  5/11/2025 (amarelo de trovoada com «fenómenos extremos de vento», laranja de chuva) — 68–100 %
  em Faro, Beja, Setúbal, Évora, Santarém, Castelo Branco e Guarda, que dariam alerta. O serviço
  vê portanto os dias de tempestades severas **generalizadas**; as trovoadas isoladas e os avisos
  subidos no próprio dia escapam-lhe, e uma frente fria muito instável pode dar um alerta onde o
  IPMA fica no amarelo.
- **Nevoeiro** (`app/alerts/fog.py`): o IPMA avisa pela **persistência** do nevoeiro (amarelo
  ≥ 48 h, laranja ≥ 72 h, vermelho ≥ 96 h). Os dados abertos do ECMWF não trazem a visibilidade
  e a grelha de 0,25° não vê os vales onde o nevoeiro persiste: no episódio laranja de
  26/12/2024–02/01/2025 (Bragança, Viseu, Guarda, Vila Real, «em especial em zonas de vale», com
  sincelo) o IFS saturava todas as noites mas aquecia e secava um pouco todas as tardes. O que
  os modelos mostram bem é o **regime** que mantém o nevoeiro, e é isso que se conta: um dia de
  nevoeiro num ponto tem o ar saturado de madrugada (temperatura − ponto de orvalho a 2 m ≤ 0,5 K
  às 06 UTC), a tarde húmida sob a inversão (≤ 3,5 K às 15 UTC), vento ≤ 3 m/s às 15 UTC e chuva
  ≤ 1 mm no dia; uma área tem um dia de nevoeiro quando um ponto o tem; a duração é 24 h × os
  dias seguidos (laranja ao 3.º dia, como o IPMA fez em 28/12/2024). Os dias antes da run vêm do
  IFS (a run das 00 UTC de cada dia, o mesmo teste). Vermelho só com 4 dias de nevoeiro que não
  levanta (≤ 1 K às 15 UTC): nunca aconteceu nos dois invernos verificados, e o IPMA manteve o
  episódio de 2024 em laranja durante oito dias. Calibração no IFS (run das 00 UTC de cada dia,
  240 dias de novembro a fevereiro de 2024/25 e 2025/26): sequências de ≥ 3 dias só no episódio
  laranja (Vila Real 8 dias, Bragança 7, Viseu 7, Guarda 4); de 2 dias, o amarelo de
  30–31/12/2025 (Bragança, Guarda, Vila Real) e quatro outras; sem o teste do vento Setúbal
  chegava a 3. Com o ensemble do ECMWF (runs das 00 UTC): para 28/12/2024, a run de 24/12 dava
  Bragança 86 %, Guarda 76 %, Vila Real 76 %, Viseu 60 %; a de 26/12, 78/62/48/28 %; a de 27/12
  (já com os dias anteriores) Bragança 92 %, Vila Real 76 % e Viseu 72 % no dia 27 — o alerta
  sairia para os quatro distritos entre 24 e 27/12; o fim do episódio (30/12–02/01) fica
  subestimado. No amarelo de 31/12/2025 a mediana era 48 h e o laranja ≤ 26 %; nos dois falsos do
  IFS (Setúbal 9/1/2025, Açores 2/12/2024) ≤ 4 %. Downloads: o ensemble só é lido quando o IFS
  mostra ≥ 3 dias seguidos numa área (com uma tarde mais tolerante, 4,5 K; nos dois invernos, só
  no episódio de 2024 e em Setúbal a 9/1/2025); então a temperatura e o ponto de orvalho das 06 e
  15 UTC de todos os membros (~130 MB por dia) e, só para os membros e dias que podem chegar às
  72 h, o vento das 15 UTC e a chuva do dia (~1 GB por análise num episódio). Modelos de apoio:
  IFS, GFS e ICON-EU (o AIFS não tem as 15 UTC), a um dia do limiar. Até 4 dias.
- **Neve** (`app/alerts/snow.py`): a grelha de 0,25° alisa as serras (a Serra da Estrela fica
  com 865 m no IFS), por isso a neve não é lida do manto de neve dos modelos mas **estimada à
  altitude de cada faixa** a partir da precipitação e da temperatura de cada ponto: temperatura
  interpolada em altura entre os 2 m (na orografia do modelo) e os 850 hPa (a 1450 m), fração de
  neve 1 a ≤ 0 °C e 0 a ≥ 2 °C, 1 cm por mm de água; o valor de um dia é a maior acumulação em
  24 h que termina nele. Cada distrito é avaliado a 1000 m (ou menos, se não tiver terreno tão
  alto), a 1600 m e acima, até à sua **altitude representativa**: a cota acima da qual tem pelo
  menos 25 km² (modelo digital de terreno Copernicus GLO-90), para que um cume de poucos
  hectares na fronteira não decida o aviso do distrito inteiro. Distritos do continente, regiões
  montanhosas da Madeira e os três grupos dos Açores (Ocidental 610 m, Central 1030 m — o Pico,
  com 28 km² acima de 1000 m —, Oriental 760 m); as costas da Madeira ficam de fora. Validado com
  o nevão de 20–21/12/2025: com a run de 19/12 00Z do ensemble, os seis distritos em laranja do
  IPMA (Viana, Braga, Vila Real, Viseu, Guarda, Castelo Branco) tinham 62–100 % e os quatro em
  amarelo (Porto, Bragança, Aveiro, Coimbra) 8–22 %.
- **Mar**: pontos 10–25 km ao largo de cada distrito costeiro, da Madeira (costa norte, costa
  sul, Porto Santo) e de cada grupo dos Açores; cada ponto lê a célula de mar mais próxima.
  Ensemble de ondas do ECMWF (`waef`, 50 membros, de 6 em 6 h, com a direção média para a regra
  do Algarve) e, como modelos determinísticos, o modelo de ondas do IFS e o GFS-Wave da NOAA.
- **Probabilidade**: os 50 membros do ensemble do ECMWF (`ef`, Open Data), lidos por intervalos
  de bytes e recortados a Portugal; uma área chega ao nível num membro quando algum dos seus
  pontos chega. Para poupar downloads, os membros só são lidos onde os produtos de
  probabilidade do ECMWF (`ep`: rajadas ≥ 90 km/h, chuva ≥ 25 mm/24 h, ondas acima de 2, 4 e
  6 m), a Tmáx/Tmín do IFS, a neve estimada pelo IFS (≥ 30 % do limiar), o índice de trovoada
  do IFS (≥ 60 % do limiar) ou os dias de nevoeiro do IFS (≥ 3 seguidos) mostram alguma hipótese.
- **Certeza** (`app/alerts/events.py`, ajustável em `config.json`): ≥ 60 % dos membros
  (vermelho ≥ 50 %), ≥ 40 % já na run anterior, metade dos modelos determinísticos (IFS, AIFS,
  GFS, ICON-EU; no mar, IFS e GFS-Wave; na neve, a 40 % do limiar; na chuva em 1 h, ICON-EU e
  GFS; na trovoada, IFS, GFS e ICON-EU a 40 % do limiar; no nevoeiro, IFS, GFS e ICON-EU a um
  dia do limiar) perto do limiar e até 4 dias de antecedência (calor e frio 6, chuva em 1 h e
  trovoada 2). Abaixo disso o
  evento fica «em vigilância», só na app. Um evento confirmado mantém-se enquanto a
  probabilidade não cair abaixo de 25 % (não há alertas e cancelamentos alternados).
- **Mensagens**: alerta (texto + 7 figuras: probabilidade, valor alto provável, membros por dia,
  pluma, evolução de run para run, modelos determinísticos, situação sinóptica), atualizações
  só quando muda o nível, as áreas ou os dias, cancelamento; os avisos laranja e vermelho do
  IPMA são reencaminhados. Das 23h às 8h chegam em silêncio, exceto os vermelhos. No Telegram,
  `/estado` lista os eventos em curso.
- **Verificação** (`app/alerts/verify.py`): na manhã seguinte (07 UTC) cada dia é verificado,
  área a área, para cada tipo de aviso do IPMA. Compara-se o que o serviço disse (alerta laranja
  ou vermelho, vigilância, alerta cancelado ou nada, e a probabilidade de laranja da run das 00
  UTC um e dois dias antes) com o **aviso do IPMA** para a mesma área e dia — a referência, porque
  o serviço existe para antecipar esses avisos — e com o **observado**: as estações do IPMA (chuva
  em 1 e 6 h, vento médio, temperatura horária) e os METAR dos aeroportos (rajadas, horas com
  visibilidade abaixo de 1 km; arquivo do Iowa Environmental Mesonet). Os dados abertos do IPMA só
  guardam as últimas 24 h de observações e os avisos em vigor, por isso o serviço arquiva os
  avisos (todos os níveis, com a hora em que apareceram) a cada ciclo e as observações de hora a
  hora, em `verif/` (400 dias). Não há observações abertas de rajadas fora dos aeroportos, de
  ondas, neve, granizo ou do nevoeiro dos vales: aí conta só o aviso do IPMA. Cada área-dia é um
  **acerto** (alerta e aviso laranja/vermelho), um **alarme falso** (alerta sem esse aviso) ou uma
  **falha** (aviso sem alerta); nos acertos, o nível (igual, abaixo, acima) e a **antecedência**
  face ao primeiro aviso laranja/vermelho no feed do IPMA. Quando um evento confirmado acaba (ou
  é cancelado) segue uma mensagem de verificação, e os avisos laranja/vermelho que nenhum alerta
  anunciou dão uma mensagem «não previsto». As pontuações acumulam (POD, FAR, antecedência,
  fiabilidade e Brier da probabilidade um dia antes) para afinar os limiares quando houver casos
  suficientes; `python -m app.alerts verificar` corre a verificação à mão. Primeiro caso
  (chuva de 21/11/2026, verificado depois): 5 acertos (Viseu, Coimbra, Leiria, Aveiro, Vila Real;
  nos três primeiros o IPMA chegou ao vermelho) e 8 falhas — seis estavam em vigilância (28–54 %),
  Lisboa e Setúbal (laranja a partir das 15 h) não tinham sinal no ensemble.
- **Separador «Avisos»** na app: estado do serviço, eventos (confirmados, em vigilância e
  anteriores), tabela por área, figuras de cada run, mensagens enviadas e a verificação (de cada
  evento e acumulada).

```bash
cd backend
../.venv/bin/python -m app.alerts configurar   # bot do Telegram (token no porta-chaves)
../.venv/bin/python -m app.alerts instalar     # agente de arranque (macOS): sempre a correr
../.venv/bin/python -m app.alerts agora --sem-envio   # um ciclo já, a mostrar o que enviaria
../.venv/bin/python -m app.alerts estado
../.venv/bin/python -m app.alerts remover      # tira o agente
```

Os dados ficam em `~/Library/Application Support/ForecastWorkstation/alerts/` (estado, runs
analisadas, figuras, registo `service.log`). O Mac tem de estar ligado e acordado para os
alertas saírem a horas.

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
