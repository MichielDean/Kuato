.PHONY: build test lint clean

build:
	go build ./...

test:
	go test ./...

lint:
	go vet ./...

clean:
	rm -f *.db