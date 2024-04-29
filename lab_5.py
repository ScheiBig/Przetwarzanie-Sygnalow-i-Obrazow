import cv2
import numpy as np
import skimage as ski  # <- na potrzeby generowania kształtów itp.
import typing as _t
import math
import itertools as it
import time
import statistics

class ansi:
	'''
	ANSI Escape Sequences - formatowanie tekstu w konsoli 
	(https://en.wikipedia.org/wiki/ANSI_escape_code)
	'''

	fg_YELLOW = "\u001b[33m"
	fg_MAGENTA = "\u001b[35m"
	fg_CYAN = "\u001b[36m"

	fgb_BLACK = "\u001b[90m"
	fgb_RED = "\u001b[91m"
	fgb_GREEN = "\u001b[92m"
	fgb_BLUE = "\u001b[94m"

	fmt_RESET = "\u001b[0m"

	clr_ALL = "\u001b[2J"

	cur_TOP_BEGIN = "\u001b[0;0H"

####################################################################################################
#                                                                                                  #
# ## Zadanie                                                                                       #
# Proszę przygotować program, który będzie śledził poruszanie się bil na stole                     #
# do snookera. Plikiem źródłowym będzie snooker2.mp4.                                              #
#                                                                                                  #
# Kroki przetwarzania:                                                                             #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# ### 1. Odfiltrowanie tła                                                                         #
# Poprzez progowanie zakresu koloru znajdź bile na stole. Kolorem bazowym będzie                   #
# kolor stołu. Punkty, których kolor różni się więcej niż tol od średniego koloru                  #
# stołu ustaw jako punkty obiektów, pozostałe - jako punkty tła. Wykonaj podobnie                  #
# jak w Lab3 - Samolot.                                                                            #
#                                                                                                  #
####################################################################################################

# Operatory do erosion i dilation
# Koła
kr_disk_3 = ski.morphology.disk(3)
kr_disk_2 = ski.morphology.disk(2)
kr_disk_1 = ski.morphology.disk(1)
# Kwadraty
kr_pixel_2 = np.ones((2, 2))
kr_pixel_1 = np.array([1])


# Maska przysłaniająca - zakrywa koszyki w stole oraz tło poza stołem,
# suplementując maskę usuwającą kolor tła
# Maska główna
out_mask = np.zeros((720, 1280), dtype=np.uint8)
# Maska do "łatania"
out_mask_middle = out_mask.copy()
# Odkrywamy powierzchnię stołu
out_mask_middle[140:-140, 210:-210] = 255
# Tworzymy koło do przykrycia koszyków
hide = (1 - ski.morphology.disk(22)) * 255
# Przykrywamy koszyki w narożnikach ...
out_mask_middle[120:165, 190:235] = hide
out_mask_middle[120:165, -235:-190] = hide
out_mask_middle[-165:-120, 190:235] = hide
out_mask_middle[-165:-120, -235:-190] = hide
# ... oraz te po środku
out_mask_middle[120:165, 618:-617] = hide
out_mask_middle[-165:-120, 618:-617] = hide
# Z łatanej maski wycinamy zakres stołu i wstawiamy do maski głównej
out_mask[140:-140, 210:-210] = out_mask_middle[140:-140, 210:-210]


class background_filter:
	'''
	Klasa służąca do usuwania tła z obrazu (tworzenia maski do usunięcia tła)
	'''

	def __init__(self, bg_cutout: np.ndarray, *, boost_dist=(0, 0, 0)) -> None:
		'''
		Inicjalizacja klasy do usuwania tła - przyjmuje wycinek tła, z którego
		buduje średni kolor tła oraz maksymalną różnicę średniego tła względem
		wycinka - te używane są później do usuwania tła z klatek nagrania.

		Opcjonalny argument [boost_dist] pozwala przesunąć maksymalną różnicę,
		co przydaje się ze względu na ciemniejsze tło w narożnikach stołu
		do bilarda
		'''

		# Wybieramy kolor idealny tła jako średnią koloru wycinka
		self.color_bg: np.ndarray = np.uint8(
			np.floor(np.mean(bg_cutout, axis=(0, 1)))
		)
		# Wybieramy próg różnicy tła jako maksymalne odchylenie wartości koloru
		# w odcinku od wartości idealnej
		diffs: np.ndarray = np.abs(np.int16(bg_cutout) - self.color_bg)
		self.max_dist: np.ndarray = diffs.max(axis=(0, 1)) \
				+ np.array(boost_dist)


	def background_removal_mask(self, frame: np.ndarray) -> np.ndarray:
		'''
		Używając parametrów tej klasy, tworzy maskę do usunięcia tła z obrazu;
		na masce kolor czarny (wartość 0) oznacza tło, biały resztę obrazu.
		'''

		# Tworzymy odwrotną maskę tła wybierając miejsca gdzie różnica koloru
		# od idealnego tła jest mniejsza od wartości progu na wszystkich kanałach
		masks = np.abs(np.int16(frame) - self.color_bg) < self.max_dist
		mask_bin = np.logical_and(
			np.logical_and(
				masks[:, :, 0],
				masks[:, :, 1],
			),
			masks[:, :, 2]
		)

		# Odwracamy maskę na oczekiwany zestaw wartości
		mask = np.uint8((1 - mask_bin) * 255)

		# Przetwarzamy czarną bilę do usunięcia artefaktów (erozja i dylatacja)
		# Głównie staramy się wyeliminować kij do bilarda
		mask = cv2.erode(mask, kr_disk_3, iterations= 1)
		mask = cv2.dilate(mask, kr_disk_1, iterations= 1)
		mask = cv2.erode(mask, kr_disk_3, iterations= 1)
		mask = cv2.dilate(mask, kr_disk_2, iterations= 1)
		mask = cv2.dilate(mask, kr_disk_2, iterations= 1)

		# Usuwamy z maski elementy poza stołem oraz koszyki
		mask = np.bitwise_and(mask, out_mask)

		return mask


####################################################################################################
#                                                                                                  #
# ### 2. Detekcja obiektów                                                                         #
# Do znalezienia bil proszę użyć obiektu                                                           #
# [SimpleBlobDetector](https://learnopencv.com/blob-detection-using-opencv-python-c/)              #
#                                                                                                  #
# Proszę ustawić odpowiednie parametry, tak aby znajdowane były obiekty                            #
# o określonym przedziale wielkości i współczynniku okrągłości. Na poniższym                       #
# rysunku widzimy obrysy znalezionych bil.                                                         #
#                                                                                                  #
####################################################################################################

# Detekcja wymaga minimalnej parametryzacji - niezbędne intensywne przetwarzanie masek bil
# sprawia że nie mamy prawie żadnych artefaktów do pomijania
def detect_balls() -> cv2.SimpleBlobDetector:
	params = cv2.SimpleBlobDetector.Params()

	params.blobColor = 255

	params.filterByConvexity = False
	params.filterByInertia = False
	params.filterByCircularity = False

	params.filterByArea = True
	params.minArea = 40

	params.minThreshold = 63
	params.maxThreshold = 127

	return cv2.SimpleBlobDetector.create(params)


####################################################################################################
#                                                                                                  #
# ### 3. Śledzenie kul                                                                             #
# Ostatnim krokiem jest śledzenie kul. Źródłem danych jest lista obiektów                          #
# znalezionych w danej ramce. Należy zbudować słownik, w którym będziemy                           #
# przechowywać dane w postaci:                                                                     #
# ```python                                                                                        #
# Kule = {                                                                                         #
# 	1: ([(x,y), (x,y), (x,y), (x,y), (x,y), (x,y), (x,y), (x,y)], False),                           #
# 	2: ([(x,y), (x,y), (x,y), (x,y), (x,y), (x,y)], True),                                          #
# 	3: ([(x,y), (x,y), (x,y), (x,y), (x,y), (x,y), (x,y), (x,y), (x,y)], True),                     #
# }                                                                                                #
# ```                                                                                              #
# gdzie kluczem jest kolejny znaleziony obiekt, a wartością lista pozycji tego                     #
# obiektu na całym filmie i znacznik aktualizacji. Klucze są inkrementowane.                       #
#                                                                                                  #
# Przypadki:                                                                                       #
# - kula się pojawia (na poprzedniej ramce w tej lokalizacji (+/- 10 pikseli)                      #
#   nie było żadnego elementu - tworzymy nowy wpis w słowniku, znacznik = True                     #
# - kula stoi (w słowniku znajduje się gdzieś ta sama pozycja (x,y) - wystarczy                    #
#   sprawdzać ostatni element listy - słownika nie aktualizujemy, znacznik = True                  #
# - kula poruszyła się (w słowniku na końcu którejś listy jest w pobliżu                           #
#   (+/-10 pikseli) element - do odpowiedniej listy dopisujemy nową pozycję,                       #
#   znacznik = True                                                                                #
# - kula zniknęła. Element słownika, który nie został zaktualizowany, otrzymuje                    #
#   znacznik False, a w następnym obiegu przenoszony jest do słownika archiwalnego.                #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

class tracking:
	'''
	Klasa służąca do przechowywania danych dotyczących śledzenia danej bili
	'''

	__ROUND = 2
	'''Dokładność z jaką wyświetlane są współrzędne (ilość miejsc po przecinku)'''

	__MAX_POS = 6
	'''Ilość wyświetlanych współrzędnych'''

	__PAD = (14 + 2 * __ROUND) * __MAX_POS + 6
	'''Szerokość linii w której wyświetlane są współrzędne'''

	__LAST = 40
	'''Liczba ostatnich punktów śledzenia, które rysowane będą na ekranie'''
	

	def __init__(self, name: str, init_pos: _t.Optional[_t.Tuple[float, float]] = None) -> None:
		'''
		:param name: Nazwa śledzonej bili
		:param init_pos: Początkowa pozycja w której znajduje się bila
		'''

		self.name = name
		'''Nazwa śledzonej bili'''

		self.trace: _t.List[_t.Union[_t.Tuple[float, float], _t.Tuple[()]]] = []
		'''
		Ścieżka bili - lista punktów w której była widziana (w trakcie poruszania).

		W przypadku gdy bila zniknie z ekranu, wstawiany jest "pusty" punkt.
		'''

		self.on_screen = False
		'''Czy bila znajduje się obecnie (w poprzedniej klatce) na ekranie'''

		self.is_moving = False
		'''Czy bila obecnie (w poprzedniej klatce) się porusza'''

		self.last_seen = 0
		'''Numer klatki w której ostatnio bila znajdowała się na ekranie'''

		self.set = True
		'''Zmienna pomocnicza do śledzenia - czy bila została przypisana do wykrytej bili'''

		if init_pos is not None:
			self.trace.append(tuple(map(lambda c: round(c, self.__ROUND), init_pos)))
			self.on_screen = True


	def last_pos(self):
		'''Zwraca ostatnią pozycję w jakiej bila znajdowała się na ekranie'''

		for t in reversed(self.trace):
			if t != ():
				return t
	

	def formatted_name(self):
		'''
		Zwraca nazwę bili z formatowaniem ANSI:
		
		* nazwa na *zielono* - bila jest w trakcie ruchu,
		* nazwa na *niebiesko* - bila stoi w miejscu,
		* nazwa na *czerwono* - bila znajduje się poza ekranem
		'''

		if not self.on_screen:
			pre = ansi.fgb_RED
		elif self.is_moving:
			pre = ansi.fgb_GREEN
		else:
			pre = ansi.fgb_BLUE
		
		return pre + self.name + ansi.fmt_RESET
	

	def trace_tail(self):
		'''
		Zwraca kilka ostatnich pozycji w których znajdowała się bila.

		Jeśli ścieżka ma więcej niż `self.__MAX_POS` pozycji, to na początku tekstu
		znajdzie się wielokropek. Tekst ścieżki jest wypełniany do maksymalnej długości
		spacjami, aby umożliwić drukowanie w konsoli w miejscu, bez czyszczenia poprzedniego
		napisu.
		'''

		if len(self.trace) > self.__MAX_POS:
			s = "..., " + str(self.trace[-self.__MAX_POS:])[1:]
		else:
			s = str(self.trace)

		return f"{s:<{self.__PAD}}"


	def append(self, point: _t.Tuple[float, float]):
		'''Dodaje pozycję na koniec ścieżki'''

		self.trace.append((round(point[0], self.__ROUND), round(point[1], self.__ROUND)))


	def remove(self):
		'''
		"Usuwa" bilę z ekranu - ustawia wartość `self.on_screen` oraz umieszcza puste współrzędne
		na koniec ścieżki
		'''

		if self.on_screen:
			self.on_screen = False
			self.trace.append(())


	def traces(self):
		'''
		Zwraca listę ścieżek w których poruszała się bila.

		Klasa wewnętrznie przechowuje "dziurę" w śledzeniu jako pusty punkt (dziurą jest
		na przykład wbicie bili do koszyka oraz wyciągnięcie jej z powrotem na stół), jednak
		na potrzeby rysowania funkcja rozdziela ścieżkę na pojedyncze ścieżki, rozdzielając
		je po pustych punktach.

		Aby nie "zaśmiecać" ekranu, lista ścieżek nie zawiera wszystkich punktów - jedynie
		`self.__LAST` ostatnich punktów.
		'''

		traces = [
			np.array(list(v), dtype=np.int32)
			# groupby po pustych punktach, stworzy naprzemiennie puste grupy (z kluczem False),
			# oraz grupy z nieprzerwanym ścieżkami (z kluczem True) - wybieramy te drugie
			for k, v in it.groupby(self.trace[-self.__LAST:], key= lambda c: c == ()) 
			if not k
		]
		return traces


def dist(fro: _t.Tuple[float, float], to: _t.Tuple[float, float]):
	'''Zwraca odległość Euklidesową między dwoma współrzędnymi'''

	return math.sqrt((fro[0] - to[0]) ** 2 + (fro[1] - to[1]) ** 2)


class balls:
	'''
	Klasa przechowująca oraz implementująca śledzenie bil na ekranie
	'''

	__DELTA_STILL = 1.0
	'''
	Odległość między dwoma pozycjami bili używana jako próg do uznania 
	bili za stojącą w miejscu
	'''


	def __init__(
		self, 
		init_white: cv2.KeyPoint,
		init_black: cv2.KeyPoint,
		init_color: _t.Sequence[cv2.KeyPoint]
	) -> None:
		'''
		:param init_white: Początkowy punkt detekcji białej bili
		:param init_black: Początkowy punkt detekcji czarnej bili
		:param init_color: Sekwencja początkowych punktów detekcji bil kolorowych
		'''

		self.white = tracking("Biała bila", init_white.pt)
		'''Śledzenie bili białej'''

		self.black = tracking("Czarna bila", init_black.pt)
		'''Śledzenie bili czarnej'''

		self.color = [tracking(f"Bila kolorowa {i + 1}", c.pt) for i, c in enumerate(init_color)]
		'''
		Śledzenie bil kolorowych. Liczba bil ustalana jest na podstawie punktów początkowych
		przekazanych w konstruktorze
		'''

		self.frame_index = 0
		'''Numer ostatnio przetwarzanej klatki'''


	def register(
		self, 
		kp_white: _t.Optional[cv2.KeyPoint],
		kp_black: _t.Optional[cv2.KeyPoint],
		kp_color: _t.Sequence[cv2.KeyPoint]
	) -> None:
		'''
		Rejestruje nowe pozycje bil, odszukując i dopasowując bile kolorowe

		:param kp_white: Punkt detekcji białej bili
		:param kp_black: Punkt detekcji czarnej bili
		:param kp_color: Sekwencja punktów detekcji kolorowych bil
		'''
		
		self.frame_index += 1

		# Rejestracja pozycji białej bili
		if kp_white is not None:
			self.white.on_screen = True
			self.white.last_seen = self.frame_index
			lp = self.white.last_pos()
			if abs(lp[0] - kp_white.pt[0]) < self.__DELTA_STILL \
					and abs(lp[1] - kp_white.pt[1]) < self.__DELTA_STILL:
				self.white.is_moving = False
			else:
				self.white.is_moving = True
				self.white.append(kp_white.pt)
		else:
			self.white.remove()
		
		# Rejestracja pozycji czarnej bili
		if kp_black is not None:
			self.black.on_screen = True
			self.black.last_seen = self.frame_index
			lp = self.black.last_pos()
			if abs(lp[0] - kp_black.pt[0]) < self.__DELTA_STILL \
					and abs(lp[1] - kp_black.pt[1]) < self.__DELTA_STILL:
				self.black.is_moving = False
			else:
				self.black.is_moving = True
				self.black.append(kp_black.pt)
		else:
			self.black.remove()

		# Wyszukanie i rejestracja bil kolorowych

		# Krok 0.1 - oznaczamy wszystkie bile jako niedopasowane
		for p in self.color:
			p.set = False

		# Krok 0.2 - tworzymy listę na "odrzucone" punkty detekcji
		skip_list: _t.Sequence[cv2.KeyPoint] = []

		# Krok 1 - przeprowadzamy detekcję bil o najwyzszym stopniu pewności.
		# "Na pewno" do bil możemy dopasować nowe pozycje, jeśli bila się nie poruszała
		# od ostatniej klatki
		for kp in kp_color:

			# Tworzymy listę bil, odległości od obecnego punktu detekcji oraz numerów bil
			# wybieramy tylko bile które znajdują się na ekranie oraz nie zostały jeszcze dopasowane
			dists = [
				(c, dist(c.last_pos(), kp.pt), i)
				for i, c in enumerate(self.color) 
				if c.on_screen and not c.set
			]
			# Sortujemy powyższą listę rosnąco, po odległości bili od punktu detekcji
			dists = sorted(dists, key=lambda d: d[1])

			# Jeśli nie mamy bil do dopasowania, to od razu odrzucamy punkt detekcji 
			# do kolejnego kroku przetwarzania
			if len(dists) == 0:
				skip_list.append(kp)
				continue

			# Jeśli najbliższa bila znajduje się poniżej progu nieporuszania się, dopasowujemy
			# do niej obecny punkt detekcji i kontynuujemy;
			# w przeciwnym wypadku odrzucamy punkt detekcji do kolejnego kroku przetwarzania
			if dists[0][1] < self.__DELTA_STILL:
				i = dists[0][2]
				self.color[i].is_moving = False
				self.color[i].on_screen = True
				self.color[i].last_seen = self.frame_index
				self.color[i].set = True
				if self.color[i].trace[-1] == ():
					self.color[i].append(kp.pt)
			else:
				skip_list.append(kp)
		
		# Jeśli dopasowaliśmy wszystkie bile, to kończymy przetwarzanie
		if len(skip_list) == 0:
			pass
		# Krok 2.A - Jeśli nie dopasowaliśmy tylko jednej bili, to mamy dwie możliwe sytuacje:
		# * tylko jedna bila się obecnie porusza,
		# * bila została wyjęta z koszyka po nielegalnym ruchu.
		elif len(skip_list) == 1:
			kp = skip_list[0]

			# Tak jak w poprzednim kroku, ponownie tworzymy rosnąco posortowana listę bil które
			# znajdują się na ekranie i nie zostały jeszcze dopasowane, oraz odległości 
			# do pozostałego punktu detekcji
			dists = [
				(c, dist(c.last_pos(), kp.pt), i) 
				for i, c in enumerate(self.color) 
				if not c.set and c.on_screen
			]
			dists = sorted(dists, key=lambda d: d[1])

			# Jeśli takie bile istnieją, do dopasowujemy punkt detekcji do najbliższej,
			# odpowiednio jako bili stojącej w miejscu, lub poruszającej się - pierwsza
			# możliwa sytuacja
			if len(dists) != 0:
				if dists[0][1] < self.__DELTA_STILL:
					i = dists[0][2]
					self.color[i].is_moving = False
					self.color[i].on_screen = True
					self.color[i].last_seen = self.frame_index
					self.color[i].set = True
					self.color[i].append(kp.pt)
				else:
					i = dists[0][2]
					self.color[i].is_moving = True
					self.color[i].on_screen = True
					self.color[i].last_seen = self.frame_index
					self.color[i].set = True
					self.color[i].append(kp.pt)
			# W przeciwnym wypadku, mamy do czynienia z drugą możliwą sytuacją - w takim wypadku
			# mamy do czynienia z "dziurą" w śledzeniu (tyczy się to zielonej bili - czasem zlewa
			# się z tłem i jej detekcja nie jest możliwa), lub z wyjęciem bili z koszyka;
			# w obydwu sytuacjach, dopasowujemy się do tej bili znajdującej się poza stołem,
			# która najkrócej się poza nim znajduje
			else:
				outer_balls = sorted(
					[c for c in self.color if not c.on_screen],
					key= lambda c: c.last_seen,
					reverse= True
				)
				outer_balls[0].is_moving = False
				outer_balls[0].on_screen = True
				outer_balls[0].last_seen = self.frame_index
				outer_balls[0].set = True
				outer_balls[0].append(kp.pt)

		# Krok 2.B - jeśli nie dopasowaliśmy więcej niż jedną bilę 
		# (najczęściej dwie), to obecnie poruszają się dwie bile kolorowe
		else:
			# Tworzymy listę możliwych dopasowań, poprzez kolejne kroki:
			# * tworzymy produkt kartezjański odrzuconych punktów detekcji oraz pozostałych bili
			#   (które nie zostały jeszcze oznaczone, oraz znajdują się obecnie na ekranie),
			# * tworzymy listę ów produktów, oraz odległości między daną bilą a punktem detekcji,
			# * sortujemy listę rosnąco po odległościach
			point_possibilities = sorted(
				map(
					lambda c: (dist(c[0].pt, c[1].last_pos()), c),
					it.product(skip_list, filter(lambda c: not c.set and c.on_screen, self.color))
				),
				key= lambda k: k[0]
			)

			# Dopóki nie wyczerpiemy wszystkich kombinacji, wykonujemy kolejne kroki:
			while len(point_possibilities) > 0:
				# * wybieramy kombinację o najmniejszej odległości,
				_, (kpp, tp) = point_possibilities[0]
				# * dopasowujemy punkt detekcji do bili z danej kombinacji,
				tp.is_moving = True
				tp.on_screen = True
				tp.last_seen = self.frame_index
				tp.set = True
				tp.append(kpp.pt)

				# * usuwamy z listy te kombinacje, które zawierają przed chwilą dopasowaną bilę
				point_possibilities = [p for p in point_possibilities if p[1][1] != tp]
			

		# Krok 3 - wszystkie pozostałe niedopasowane bile "usuwamy z ekranu"
		for p in self.color:
			if not p.set:
				p.is_moving = False
				p.remove()
	

	# Poniżej znajduje się logika "kolorowego" drukowania stanu śledzenia - nie jest
	# istotna dla działania samego programu

	def __trace_pre(self, fo: tracking):
		'''
		Zwraca prefix z kolorem czcionki ANSI dla śledzenia `fo`.

		Jeśli śledzenie dotyczy bili która od dłuższego czasu przebywa poza ekranem,
		to lista punktów jej ścieżki w konsoli drukowana jest na szaro.
		'''

		if fo.last_seen + 50 < self.frame_index:
			return ansi.fgb_BLACK
		else:
			return ""


	def __str__(self) -> str:
		'''
		Drukuje sformatowaną (ANSI) reprezentację stanu śledzenia:
		* nazwy bil, pokolorowane w zależności od stanu bili:
		  * zielony - bila się porusza,
		  * niebieski - bila stoi w miejscu,
		  * czerwony - bila znajduje się poza stołem,
		* skróconą listę punktów ścieżki:
		  * lista drukowana jest na szaro, jeśli bila od dłuższego czasu przebywa poza ekranem.
		'''

		lines = list()
		
		lines.append(self.white.formatted_name()+ ":")
		lines.append("  " + self.__trace_pre(self.white) + self.white.trace_tail() + ansi.fmt_RESET)
		
		lines.append(self.black.formatted_name() + ":")
		lines.append("  " + self.__trace_pre(self.black) + self.black.trace_tail() + ansi.fmt_RESET)
		
		for ball in self.color:
			lines.append(ball.formatted_name() + ":")
			lines.append("  " + self.__trace_pre(ball) + ball.trace_tail() + ansi.fmt_RESET)

		return "\n".join(lines)



####################################################################################################
#                                                                                                  #
# ## Główna część programu:                                                                        #
#                                                                                                  #
####################################################################################################


cap = cv2.VideoCapture("lab5/Snooker2.mp4")

keep = True
'''Czy kontynuować odtwarzanie video'''

init = True
'''Czy przeprowadzić inicjalizację zasobów'''

colors = [ cv2.cvtColor(
	np.reshape(np.array([6 + i * 13, 127, 255], dtype=np.uint8), (1, 1, 3)), cv2.COLOR_HSV2BGR
).reshape(3) for i in range(14) ]
'''Lista kolorów wygenerowanych proceduralnie, używana do rysowania ścieżek i etykiet kolorowych bil'''
colors = [(float(c[0]), float(c[1]), float(c[2])) for c in colors]

bg_filter: background_filter
'''Obiekt do usuwania tła (sukna stołu)'''

detector = detect_balls()
'''Obiekt do wykrywania bil'''

# Czyszczenie konsoli
print(ansi.clr_ALL, end= None)

ballz: balls
'''Obiekt śledzenia bil'''

fps: list[float] = []
'''Lista FPSów (do liczenia średniej)'''

while keep:
	# Drukowanie danych czasowych / pozycyjnych
	if not init:
		if ballz.frame_index == 0:
			print(ansi.clr_ALL, end=None)
		t = time.time() - t
		print(
			f"{ansi.cur_TOP_BEGIN + ansi.fg_YELLOW}Przetwarzanie poprzedniej klatki:" 
			f"{ansi.fmt_RESET} {t:5.3f}sekund, {ansi.fg_MAGENTA}FPS:{ansi.fmt_RESET} {1/t:6.3f}"
		)
		fps.append(1/t)
		print(
			f"{ansi.fg_CYAN}Klatka numer:{ansi.fmt_RESET} {int(pos_frame)}, "
			f"{ansi.fg_MAGENTA}Średnie FPS:{ansi.fmt_RESET} {statistics.fmean(fps):6.3f}"
		)
	t = time.time()
	flag, frame = cap.read()
	if flag:
		# Inicjalizacja - pobranie wycinka stołu z pierwszej klatki i utworzenie z jego pomocą
		# obiektu do usuwania tła
		if init:
			h, w, _ = frame.shape
			off_h = int((h - 320) / 2)
			off_w = int((w - 320) / 2)
			fr_cut = frame[
				off_h: -off_h,
				off_w: -off_w
			]
			bg_filter = background_filter(fr_cut, boost_dist=(5, 15, 5))

		pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

		# Utworzenie maski tła
		mask = bg_filter.background_removal_mask(frame)

		# Klatka w skali HSL - pozwoli podzielić bile w zależności od koloru
		frame_det = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

		# Robimy progowanie obrazu bili białej oraz czarnej po kanale *Lightness* 
		frame_white_ball = np.uint8((frame_det[:, :, 1] > 190) * 255)
		frame_black_ball = np.uint8((frame_det[:, :, 1] < 40) * 255)

		# Przetwarzamy białą bilę do usunięcia artefaktów (erozja i dylatacja)
		frame_white_ball = cv2.erode(frame_white_ball, kr_disk_2, iterations= 1)	
		frame_white_ball = cv2.dilate(frame_white_ball, kr_disk_1, iterations= 1)	
		frame_white_ball = cv2.erode(frame_white_ball, kr_disk_2, iterations= 1)	
		frame_white_ball = cv2.dilate(frame_white_ball, kr_disk_2, iterations= 1)	

		# Przetwarzamy czarną bilę do usunięcia artefaktów (erozja i dylatacja)
		frame_black_ball = cv2.dilate(frame_black_ball, kr_pixel_2, iterations= 1)
		frame_black_ball = cv2.erode(frame_black_ball, kr_pixel_1, iterations= 1)
		frame_black_ball = cv2.dilate(frame_black_ball, kr_disk_2, iterations= 1)

		# Tworzymy obraz bil kolorowych jako "wszystko oprócz bili białej i czarnej"
		frame_color_ball = np.uint8(np.logical_not(
			np.logical_or(frame_white_ball, frame_black_ball)
		) * 255)  # kolorowe bile

		# Nakładamy maskę do usuwania tła na obrazy z bilami
		frame_white_ball = np.bitwise_and(frame_white_ball, mask)
		frame_black_ball = np.bitwise_and(frame_black_ball, mask)
		frame_color_ball = np.bitwise_and(frame_color_ball, mask)

		# Przetwarzamy kolorowe bile do usunięcia artefaktów (erozja i dylatacja)
		frame_color_ball = cv2.erode(frame_color_ball, kr_disk_3, iterations= 1)
		frame_color_ball = cv2.dilate(frame_color_ball, kr_disk_2, iterations= 1)
		frame_color_ball = cv2.dilate(frame_color_ball, kr_disk_3, iterations= 1)
		
		# Ponownie nakładamy maskę do usuwania tła na obrazy z kolorowymi bilami
		# aby usunąć nadmiar dylatacji
		frame_color_ball = np.bitwise_and(frame_color_ball, mask)

		# Wykrywamy bile na obrazach
		kpt_white = detector.detect(frame_white_ball)
		kpt_black = detector.detect(frame_black_ball)
		kpt_color = detector.detect(frame_color_ball)

		# Przekazujemy wykryte bile do przetwarzania
		if init:
			ballz = balls(kpt_white[0], kpt_black[0], kpt_color)
			init = False
		else:
			ballz.register(
				kpt_white[0] if len(kpt_white) > 0 else None,
				kpt_black[0] if len(kpt_black) > 0 else None,
				kpt_color
			)
		init = False
		
		# Drukujemy status śledzenia do konsoli
		######## UWAGA ! ########
		# Do poprawnego drukowania panelu statusowego wymagane jest okno konsoli
		# o wymiarach minimalnych 34 wiersze i 114 kolumn znaków
		print(ballz, sep= None)

		# Tworzymy planszę do wyświetlania wykrytych plam oraz ścieżek śledzenia
		mask[np.bool_(255 - out_mask)] = 31
		mask[mask == 255] = 127
		mask[frame_white_ball == 255] = 192
		mask[frame_black_ball == 255] = 63
  
		# Rysujemy na planszy wykryte bile:
		# * żółty - bila biała
		mask = cv2.drawKeypoints(
			mask, kpt_white, np.array([]), (0, 191, 191),
			flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
		)
		# * błękitny - bila biała
		mask = cv2.drawKeypoints(
			mask, kpt_black, np.array([]), (191, 191, 0),
			flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
		)
		# * czerwony - bile kolorowe (wszystkie)
		mask = cv2.drawKeypoints(
			mask, kpt_color, np.array([]), (31, 31, 255),
			flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
		)
  
		# Rysujemy na planszy ścieżki śledzonych bil (poza białą bilą, rysujemy
		# tylko ścieżki bil znajdujących się obecnie na ekranie), oraz dodajemy
		# etykiety na ekranie
		# * biały - bila biała
		mask = cv2.polylines(
				mask, 
				ballz.white.traces(),
				False, 
				(255, 255, 255)
		)
		lp = ballz.white.last_pos()
		mask = cv2.putText(
			mask,
			"B",
			(int(lp[0] + 5), int(lp[1])),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255)
		)
		# * szary - bila czarna
		if ballz.black.on_screen:
			mask = cv2.polylines(
				mask,
				ballz.black.traces(),
				False, 
				(95, 95, 95)
			)
			lp = ballz.black.last_pos()
			mask = cv2.putText(
				mask,
				"C",
				(int(lp[0] + 5), int(lp[1])),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(95, 95, 95)
			)
		# * kolejne barwy - bila kolorowe (kolory ścieżek niepowiązane z kolorami
		# danych bil)
		for i, c in enumerate(ballz.color):
			if c.on_screen:
				mask = cv2.polylines(
					mask,
					c.traces(),
					False,
					colors[i]
				)
				lp = c.last_pos()
				mask = cv2.putText(
					mask,
					c.name.split(" ")[-1],
					(int(lp[0] + 5), int(lp[1])),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					colors[i]
				)

		# Wyświetlamy surową klatkę oraz planszę z detekcją / śledzeniem
		cv2.imshow("Frame", frame)
		cv2.imshow("Ball detection", mask)

	else:
		# Jeśli nie pobrano kolejnej klatki - nagranie się zakończyło;
		# kończymy program
		cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
		print("Video finished")
		keep = False
		cv2.destroyAllWindows()
		cap.release()

	# Kończymy natychmiast, jeśli wciśnięty zostanie [Esc]
	if cv2.waitKey(1) == 27:
		keep = False
		cv2.destroyAllWindows()
		cap.release()
		break
