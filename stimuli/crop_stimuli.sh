# Crop images
convert raw/0.jpg -crop 250x250+275+60 cropped/0.jpg
convert raw/1.jpg -crop 420x420+100+2  cropped/1.jpg
convert raw/2.jpg -crop 360x360+130+5  cropped/2.jpg
convert raw/3.jpg -crop 440x440+20+20  cropped/3.jpg
convert raw/4.jpg -crop 360x360+120+80 cropped/4.jpg
convert raw/5.jpg -crop 460x460+100+5  cropped/5.jpg

for filename in cropped/*.jpg; do
    convert $filename -resize 256x256! $filename # Resize to a common size
    convert $filename -normalize "${filename}" # Normalize brightness histograms
done

# Print info to check that it worked
identify cropped/*.jpg
