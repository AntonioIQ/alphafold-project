"""
CLI principal del proyecto AlphaFold Comparison.

Uso:
    python -m alphafold_comparison <command> [options]

Comandos:
    download    Descargar estructuras de PDB y AlphaFold
    process     Procesar y alinear pares de estructuras
    validate    Validar correspondencia de pares proteicos
    analyze     Ejecutar análisis estructural
    config      Mostrar configuración actual
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift args para el sub-parser

    if command == "download":
        from alphafold_comparison.download.downloader import main as download_main
        return download_main()

    elif command == "process":
        from alphafold_comparison.preprocessing.processor import main as process_main
        return process_main()

    elif command == "validate":
        from alphafold_comparison.preprocessing.validator import main as validate_main
        return validate_main()

    elif command == "analyze":
        from alphafold_comparison.analysis.structural import main as analyze_main
        return analyze_main()

    elif command == "config":
        from alphafold_comparison.config import Config
        Config.summary()
        return 0

    else:
        print(f"Comando desconocido: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
