all:

prepare_dataset:
	python dataset.py

pdf:
	pandoc README.md -o README.pdf
	$(MAKE) -C ./reports pdf

cleanall:
	rm -f README.pdf
	$(MAKE) -C ./reports clean
	$(MAKE) -C ./Hungarian-Algorithm-in-C-Language clean
	rm -f main.o

compile_c:
	$(MAKE) -C ./Hungarian-Algorithm-in-C-Language

.PHONY: prepare_dataset pdf cleanall
