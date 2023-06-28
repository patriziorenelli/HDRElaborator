import cv2
import numpy

class HDRElaborator:

    # kernel gaussiano comune 
    kernel = cv2.getGaussianKernel(ksize=5, sigma=-1)

    # Costruttore
    def __init__(self, imagesName, layers):
        # Nomi delle immagini su cui si vuole applicare l'algoritmo per calcolare l'immagini in HDR 
        self.imagesName = imagesName
        # Livelli usati per le piramidi 
        self.layers = layers
        # Array che conterrà le varie immagini
        self.images = []
        self.alignedImage = None


    # Carica le immagini indicate nel main
    def loadImage(self):
        # Carico le singole immagini richieste 
        for im in self.imagesName:
            image = None
            image = cv2.imread('./Immagini/'+ im)
            # Controllo che l'immagine sia stata effettivamente caricata 
            if image is not None:
                self.images.append(image)
            else:
                print("Immagine {} non trovata".format('test/'+ im))
        # Se ho meno di 2 immagini è inutile andare avanti
        if len(self.images)  < 2:
            return    

        interp = None
        # Calcolo la dimensione media delle foto 
        dim = [0,0]
        for im in self.images:
            dim = numpy.add(dim, im.shape[:2])

        # Dimensione media dei valori di w e h delle future foto
        dim = dim / len(self.images)

        dimf = [0,0]
        # Controllo se le dimensioni sono compatibili con le future resize, se lo sono le uso oppure calcolo i nuovi valori 
        if int(dim[0]) % (2**self.layers) != 0:
            dimf[0] = dim[0] + int((2**self.layers) / 2)
            dimf[0] = dimf[0] - ( dimf[0] % (2**self.layers)) 
        else:
            dimf[0] = dim[0]

        if  int(dim[1]) % (2**self.layers) != 0:
            dimf[1] = dim[1] + int((2**self.layers) / 2)
            dimf[1] = dimf[1] - ( dimf[1] % (2**self.layers)) 
        else:
            dimf[1] = dim[1]

        # cast in interi
        dimf = [int(x) for x in dimf]

        '''
        Rendo le immagini tutte della stessa dimensione in modo che siano compatibili alle future resize,
        durante le varie resize si andra' ad arrotondare i risultati in interi, nel passaggio da una resize che rende 
        l'immagine più piccolaad una resize che la rende più grande potrei perdere dei pixel e l'algoritmo andrebbe in errore nel momento che 
        cerca di sommare o moltiplicare 2 matrici di dimensioni differenti
        '''
        for x in range(0, len(self.images)):
            sh, sw = ( dimf[0], dimf[1] )
            h, w = self.images[x].shape[:2]
            # Scelgo l'interpolazione migliore a secondo se si stanno decrementando o aumentando le dimensioni delle foto 
            if h > sh or w > sw:
                interp = cv2.INTER_AREA 
            else: 
                interp = cv2.INTER_CUBIC
            self.images[x] = cv2.resize(self.images[x], ( dimf[1],  dimf[0] ), interpolation  = interp)

    
    # Effettua un allineamento soft delle immagini passate
    def align(self):
        alignMTB = cv2.createAlignMTB()
        self.alignedImage = self.images
        alignMTB.process(self.images, self.alignedImage)

    # Calcola il peso delle immagini passate in input, andando a pesare il contrasto, la saturazione e l'esposizione 
    def findImportance(self, images):
        # Importance rappresenta l'importanza che daremo alle varie immagini del set che si vuole fondere  
        importance = []
        # Costruisco una matrice della dimensione (h*w) delle singole foto inizializzata tutta a 0 
        singleImportance = numpy.zeros(images[0].shape[:2], dtype=numpy.float32)
        # Analizzo ogni immagine separatamente per calcolarne l'importanza 
        for ima in images:
            # Divido tutti i valori dell'immagine in modo da normalizzarne i valori nel range [0,1]
            im = numpy.float32(ima)/255
            # Costruisco una matrice della dimensione (h*w) delle singole foto inizializzata tutta a 0 
            weight = numpy.zeros(images[0].shape[:2], dtype=numpy.float32)
            '''
            Per calcolare il valore relativo al peso del contrasto
            Applichiamo un filtro Laplaciano ad ogni immagine, che è stata precedentemente convertita in scala di grigi, prendendo il risultato del filtro in valore assoluto.
            Con cv2.CV_32F impostiamo la profondità dell'immagine di destinazione. 
            '''
            contrast = numpy.absolute(cv2.Laplacian( cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F)) + 0.5
            # Calcoliamo la moltiplicazione tra le due matrici, i pesi originali e quelli relativi al contrasto
            weight = numpy.add(weight, contrast)

            '''
            Per calcolare il valore relativo al peso della saturazione
            Calcoliamo la devizione per ciascun pixel dell'immagine
            '''
            weight = numpy.add(weight, numpy.std(im, axis=2, dtype=numpy.float32) + 0.5 )

            '''
            Per calcolare il valore relativo al peso dell'esposizione
            Andiamo a pesare ogni intensità su una campana di Gauss
            Applichiamo la campana di Gauss singolarmente ad ogni canale, moltiplicando tra loro i vari risulti ed ottenere la vera esposizione
            '''
            # uso numpy.prod per calcolare il prodotto della matrice restituita

            # Analizzo solo il canale R
            imR = im.copy()
            imR[:,:,0] = 0
            imR[:,:,1] = 0
            exposure = numpy.prod( numpy.exp(-((imR - 0.5)**2)/(2*(0.2**2))), axis=2, dtype=numpy.float32)  + 0.5

            # Analizzo solo il canale B
            imB = im.copy()
            imB[:,:,1] = 0
            imB[:,:,2] = 0
            exposure1 = numpy.prod( numpy.exp(-((imB - 0.5)**2)/(2*(0.2**2))), axis=2, dtype=numpy.float32)  + 0.5

            # Analizzo solo il canale G
            imG = im.copy()
            imG[:,:,0] = 0
            imG[:,:,2] = 0
            exposure2 = numpy.prod( numpy.exp(-((imG - 0.5)**2)/(2*(0.2**2))), axis=2, dtype=numpy.float32)  + 0.5

            exposure = numpy.multiply(exposure, exposure1)
            exposure = numpy.multiply(exposure, exposure2)
            weight = numpy.add(weight, exposure)
            singleImportance += weight
            
            importance.append(weight)

        for x in range(0, len(importance)):
            
            importance[x][singleImportance > 0] = importance[x][singleImportance > 0] / singleImportance[singleImportance > 0]
            importance[x] = numpy.uint8(importance[x]*255)
        return importance

    
    # Funzione che si occupa di effettuare sia il resize che l'applicazione di un filtro passo basso
    def lowPassFilter(self, image, exp):
        if exp:
            # Ritorno l'immagine filtrata di dimensione ridotta
            return cv2.resize(cv2.filter2D(image, -1, HDRElaborator.kernel ), None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA )
        else:
            # Ritorno l'immagine filtrata di dimensione maggiore 
            return cv2.resize(  cv2.filter2D(image, -1, HDRElaborator.kernel ), None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC )
           
    
    # Calcola la piramide Gaussiana relativa all'immagine passata.
    def gaussPyramid(self, img, layers):
        cpImage = img.copy()
        gPyramid = [cpImage]
        for x in range(0,layers):
            cpImage = self.lowPassFilter(cpImage,True)
            gPyramid.append(cpImage)
        return gPyramid

    # Calcola la piramide Laplanica relativa all'immagine passata.
    def laplPyramid(self, img ):
        gPyramid = self.gaussPyramid(img, self.layers+1)

        lPyramid = [gPyramid[self.layers-1]]
        for x in range(self.layers-1, 0, -1):
            workedImage = self.lowPassFilter(gPyramid[x], False)
            subImage = cv2.subtract(gPyramid[x-1], workedImage)
            lPyramid = [subImage] + lPyramid
        return lPyramid
        

    def overlap(self, images):
        # Calcolo i pesi delle singole immagini, andando a pesare esposizione, saturazione e contrasto 
        importance = self.findImportance(images)
        lapl, gaus = [], []

        # Calcolo le piramidi laplamiche e gaussiane relative alle immagine e alle mappa pesate 
        for x in range(0, len(importance)):
            lapl.append(self.laplPyramid(images[x]))
            gaus.append(self.gaussPyramid(importance[x], self.layers))
        

        # Combina le piramidi con i pesi  
        pyr = []
        for x in range(0,self.layers):
            # Costruisco una matrice della dimensione (h*w) delle singole foto inizializzata tutta a 0
            moreDetails = numpy.zeros(lapl[0][x].shape, dtype=numpy.uint8)

            # per ogni immagine che si vuole fondere nell'HDR 
            for y in range(0, len(images)):
                # Prende le varie immagini laplaniche nelle varie dimensioni ogni volta 
                laplImage = lapl[y][x]
                
                '''
                Prendo le singole immagini gaussiane nelle varie dimensioni
                Trasformo i valore delle singole immagini in float32 e 
                li divido per 255 in modo da normalizzare i valori in un range [0,1]
                '''
                gausNormalized = numpy.float32(gaus[y][x]) / 255
                stackGauss = numpy.dstack((gausNormalized, gausNormalized, gausNormalized))

                '''
                Moltiplico tra loro le immagini relative ai singoli livelli delle piramidi
                Come input gli forniamo la singola immagine laplanica, l'immagine gaussiana manipolata e 
                la profonditò della matrice cv2.CV_8UC3 (matrice intera senza segno a 8 bit con 3 canali)
                '''
                multiImage = cv2.multiply(laplImage,stackGauss, dtype=cv2.CV_8UC3)

                '''
                Sommo l'immagine generata dalla multiply alla matrice precedentemente inizializzata, 
                in questo modo vado a catturare maggiori dettagli da ogni foto analizzata 
                '''
                moreDetails = cv2.add(moreDetails, multiImage)
            # Aggiungo le varie immagini generate, una per ogni livello analizzato
            pyr.append(moreDetails)


        fused = pyr[len(pyr)-1]  
        for i in range( len(pyr)-2, -1, -1):
            fused = cv2.add( pyr[i], self.lowPassFilter(fused,False))

        return fused


    def getHDRImage(self):
        self.loadImage()
        # Controllo che almeno 2 immagini siano state caricate ( l'esecuzione su una sola immagine è inutile)
        if len(self.images) >= 2:
            self.align()
            result = self.overlap(self.alignedImage)
            cv2.imwrite('Immagini/RisultatoFusione.jpg', result)
            print("Immagine HDR creata")
        else:
            print("Set di immagini non valide per l'esecuzione dell'algoritmo")


if __name__ == "__main__":
    
# Set di immagini di test presenti nella cartella Immagini 
# ['monument_opt.jpg', 'monument_over.jpg', 'monument_under.jpg' ]
# ['casa1.jpg', 'casa3.jpg', 'casa2.jpg']
# ['venezia1.jpg', 'venezia2.jpg', 'venezia3.jpg' ]
# ['finA.jpg', 'finB.jpg', 'finC.jpg', 'finD.jpg']
# [ 'pig2.jpg', 'pig3.jpg', 'pig4.jpg']
# [ 'pig1.jpg', 'pig2.jpg', 'pig3.jpg', 'pig4.jpg']

    hdr = HDRElaborator(  ['venezia1.jpg', 'venezia2.jpg', 'venezia3.jpg' ], 2)
    hdr.getHDRImage()
