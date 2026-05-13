

VEGETATION_INDEX = {# rgb bands
'grvi': '(green - red)/(green + red)',
'grvi_eq': '(green_eq - red_eq)/(green_eq + red_eq)',
'mgrvi': '((green*green) - (red*red))/((green*green) + (red*red))',
'rgbvi': '((green*green) - (blue*red))/ ((green*green) + (blue*red))',
 # nir indexes
'ndvi': '(nir - red)/(nir + red)',
'ndre': '(nir - edge)/(nir + edge)',
'gndvi': '(nir - green)/(nir + green)',
'regnvi': '(edge - green)/(edge + green)',
'reci': '(nir / edge) - 1',
'evi': '2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1))',
'negvi': '((nir*nir) - (edge*green))/ ((nir*nir) + (edge*green))'}
