#!/usr/bin/env python3

import sys


def main():
    models = []
    with open("admin/models.txt") as models_fp:
        for l in models_fp:
            models.append(l.strip())

    print("version: '3.8'")

    volumes = []

    print("services:")

    for model in models:
        for service in [f"{model:s}-dev", f"{model:s}-prod"]:
            print(f"  {service:s}:")
            print(f"    build: .")
            print(f"    secrets:")

            if service.endswith("-prod"):
                print(f"    - target: model_id")
                print(f"      source: {model:s}-model_id")

            print(f"    - target: model_name")
            print(f"      source: {model}-model_name")
            print(f"    - target: public_id")
            print(f"      source: public_id")
            print(f"    - target: secret_key")
            print(f"      source: secret_key")
            print(f"    volumes:")
            print(f"    - data-{service:s}:/data")
            print(f"    - data-shared:/data-shared")

            if service.endswith("-prod"):
                print(f"    - data-submissions:/data-submissions")

            volumes.append(f"data-{service:s}")

    print("volumes:")
    volumes.append("data-shared")
    volumes.append("data-submissions")
    for volume in volumes:
        print(f"  {volume:s}:")

    print("secrets:")
    for model in models:
        print(f"  {model:s}-model_id:")
        print(f"    file: secrets/{model:s}-model_id.txt")
        print(f"  {model:s}-model_name:")
        print(f"    file: secrets/{model:s}-model_name.txt")

    print(f"  public_id:")
    print(f"    file: secrets/public_id.txt")
    print(f"  secret_key:")
    print(f"    file: secrets/secret_key.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
