# HDRElaborator
## General Description
  Project created for the exam of the Computer Graphics course of the degree in computer science of the "La Sapienza University of Rome".
 ## More details
  The project tries to implement the algorithm in the paper "Exposure Fusion of Tom Mertens, Jan Kautz and Frank Van Reeth", which illustrates one of the possible techniques for superimposing images with different    exposures, saturation and contrast. Exploit Gaussian and Laplanic pyramids. My implementation takes advantage of a Python class that implements all the necessary functions. For more information on my implementation and the results obtained, consult the present documentation.
  I'm sorry but the original documentation and comments in the code were written in Italian. The English version of the documentation has been translated automatically, I do not guarantee its correctness.

  ## How to run 
  To run the algorithm, the OpenCV (cv2) and numpy libraries must be installed
  To execute the algorithm it will be sufficient to move the own images to be processed in the Immagini folder of the project, after which in the main section create an object of the type of the HDRElaborator class, passing it as parameters the names of the images and the number of levels to be processed. Now you just need to call the getHDRImage method (this call is already present in the main). The algorithm will create as a final result the image 'ResultFusion.jpg' in the Immagini folder
  
  
