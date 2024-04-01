
ENTRY(Reset_Handler)

HEAP_SIZE = DEFINED(__heap_size__) ? __heap_size__ : 0x0400;
STACK_SIZE = 0x2000;

RTT_HEAP_SIZE = 128*1024;
OMV_FB_TOTAL_SIZE = (10*1024*1024);
OMV_MAIN_FB_SIZE = (4*720*1280);
OMV_FB_ALLOC_SIZE = (OMV_FB_TOTAL_SIZE - OMV_MAIN_FB_SIZE);
OMV_FB_ALLOC_FAST_SIZE = 296*1024;


MEMORY
{
  m_boot_data (RX) : ORIGIN = 0x60000000, LENGTH = 0x00001000
  m_image_vertor_table (RX) : ORIGIN = 0x60001000, LENGTH = 0x00001000

  m_interrupts (RX) : ORIGIN = 0x60002000, LENGTH = 0x00000400
  m_text (RX) : ORIGIN = 0x60002400, LENGTH = 0x007FDC00

  m_itcm (RW) : ORIGIN = 0x00000000, LENGTH = 0x00008000
  m_dtcm (RW) : ORIGIN = 0x20000000, LENGTH = 0x00078000
  m_ocram (RW) : ORIGIN = 0x20200000, LENGTH = 0x00080000

  m_sdram (RW) : ORIGIN = 0x80000000, LENGTH = 0x01E00000
  m_nocache (RW) : ORIGIN = 0x81E00000, LENGTH = 0x00200000
}


SECTIONS
{
  .boot_data :
  {
    . = ALIGN(4);
    __FLASH_BASE = .;
    KEEP(* (.boot_hdr.conf))
    . = ALIGN(4);
  } > m_boot_data

  ivt_begin= ORIGIN(m_boot_data) + LENGTH(m_boot_data);

  .image_vertor_table : AT(ivt_begin)
  {
    . = ALIGN(4);
    KEEP(*(.boot_hdr.ivt))
    KEEP(*(.boot_hdr.boot_data))
    KEEP(*(.boot_hdr.dcd_data))
    . = ALIGN(4);
  } > m_image_vertor_table


  .interrupts :
  {
    __VECTOR_TABLE = .;
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > m_interrupts

  __VECTOR_RAM = __VECTOR_TABLE;
  __RAM_VECTOR_TABLE_SIZE_BYTES = 0x0;


  .text :
  {
    . = ALIGN(4);
    KEEP (*(EXCLUDE_FILE(*gc.* *xalloc.* *collections.* *rt_mem*).text .rodata* .constdata*))







    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))


    . = ALIGN(32);
    __fsymtab_start = .;
    KEEP(*(FSymTab))
    __fsymtab_end = .;
    . = ALIGN(4);
    __vsymtab_start = .;
    KEEP(*(VSymTab))
    __vsymtab_end = .;
    . = ALIGN(4);


    . = ALIGN(4);
    __rt_init_start = .;
    KEEP(*(SORT(.rti_fn*)))
    __rt_init_end = .;
  } > m_text

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > m_text

  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } > m_text

 .ctors :
  {
    PROVIDE(__ctors_start__ = .);

    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))





    KEEP (*(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))

    PROVIDE(__ctors_end__ = .);
  } > m_text

  .dtors :
  {
    PROVIDE(__dtors_start__ = .);

    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))

    PROVIDE(__dtors_end__ = .);
  } > m_text

  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > m_text

  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > m_text

  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > m_text

  __etext = .;
  __DATA_ROM = .;

   .data : AT(__DATA_ROM)
  {
    __DATA_RAM = .;
    __data_start__ = .;

    KEEP (*(EXCLUDE_FILE(*/lv* *usb*).data .data*))

    *(.dmaFramebuffer)
    *(NonCacheable.init)
    *(NonCacheable)
    * (.jpeg_buf)
    . = ALIGN(4);
    __data_end__ = .;
  } > m_dtcm

  __DATA_ROM_END = __DATA_ROM + (__data_end__ - __data_start__);
  .ram_code : AT (__DATA_ROM_END)
  {
    . = ALIGN(4);
    __ram_function_start__ = .;
    KEEP (*gc.*(.text* .rodata*))

    KEEP (*rt_mem*(.text* .rodata*))
    KEEP (*collections.*(.text* .rodata*))
    KEEP (*array.*(.text* .rodata*))
    KEEP (*xalloc.*(.text* .rodata*))
    KEEP(* (.ram_code))
     __ram_function_end__ = .;
  } > m_itcm
  __RAMCODE_ROM_END = __DATA_ROM_END + (__ram_function_end__ - __ram_function_start__);

  .lv_data : AT (__RAMCODE_ROM_END)
  {
    PROVIDE(_lv_data_start = .) ;
    KEEP(*/lv*(.data* ))
    PROVIDE(_lv_data_end = .);
    PROVIDE(_lv_bss_start = .) ;
    KEEP(*/lv*(.bss*))
    PROVIDE(_lv_bss_end = .) ;
  } > m_dtcm

  __ZI_ROM = __RAMCODE_ROM_END + (_lv_data_end - _lv_data_start);

  .bss : AT (__ZI_ROM)
  {
    . = ALIGN(4);
    __START_BSS = .;
    __bss_start__ = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
    __END_BSS = .;
  } > m_dtcm

  _BSS_ROM_END = __ZI_ROM +(__END_BSS - __START_BSS);

  .dtat_dtcm : AT (_BSS_ROM_END)
  {
    . = ALIGN(32);
    WEIT_CACHE_AREA = .;
    WEIT_CACHE_AREA_END = WEIT_CACHE_AREA + 61440;
    __NDATA_ROM = WEIT_CACHE_AREA_END;
      __noncachedata_start__ = .;
    . = ALIGN(4);
    __noncachedata_init_end__ = .;
    . = ALIGN(4);
    __noncachedata_end__ = .;
    . = ALIGN(32);
    __data_buf_end = .;
  } > m_dtcm


  .fb_alloc_fast : AT (_BSS_ROM_END)
  {
    OMV_FB_ALLOC = .;
    PROVIDE(_fballoc_overlay_start = .);
    . += OMV_FB_ALLOC_FAST_SIZE;
    PROVIDE(_fballoc_overlay_end = .);
  } > m_dtcm

  _DTCM_ROM_END = _BSS_ROM_END;

  .stack : AT (_DTCM_ROM_END)
  {
    . = ALIGN(32);
    stack_start = .;
    . += STACK_SIZE;
    stack_end = .;
    __StackTop = .;
  } > m_sdram

  .heap : AT (_DTCM_ROM_END)
    {
        _pvHeapStart = .;
        . += HEAP_SIZE;
        . = ALIGN(4);
        _pvHeapLimit = .;
    } > m_sdram

  .ER_LCD_FB : AT (_DTCM_ROM_END)
  {
    _lcd_fb_start = .;
    KEEP(* (.lcd_fb))
    PROVIDE(_lcd_fb_end = .);
  } > m_sdram

  _SDRAM_ROM_LCD_END = _DTCM_ROM_END + (_lcd_fb_end - _lcd_fb_start);

  .ocram_eara : AT (_SDRAM_ROM_LCD_END)
  {
    OCRAM_AREA = .;
 .+ = 0x00080000;
 OCRAM_AREA_END = .;
  } > m_ocram

  .RTT_HEAP : AT (_SDRAM_ROM_LCD_END)
  {
    PROVIDE(heap_start = .);
    . = ALIGN(32);
    . += RTT_HEAP_SIZE;
    heap_end = .;

    RTT_MPY_THREAD_STACK = .;
    . += 327680;
    _RTT_MPY_THREAD_STACK_END = .;

    MPY_HEAP_START = .;
    . += 6291456;
    _MPY_HEAP_START_END = .;

    OMV_MAIN_FB = .;
    . += OMV_MAIN_FB_SIZE + (2*1024*1024);
    OMV_FB_END = .;
    . += OMV_FB_ALLOC_SIZE + (2*1024*1024);
    OMV_FB_ALLOC = .;
    _fballoc = .;
  } > m_sdram

  .RW_SRAM_NCACHE : AT (_SDRAM_ROM_LCD_END)
  {
    . = ALIGN(32);
    .sdram_ncache_start = .;
    * (.usb_buf)
    *(*m_usb_dma_init_data)
    *(*m_usb_dma_noninit_data)
    .sdram_ncache_end = .;
  } > m_nocache

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
